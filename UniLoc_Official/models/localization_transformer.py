
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .old.localization_transformer import (
    LocalizationLoss,
    PreNormEncoderLayer,
    SinusoidalPositionalEncoding,
)


class LocalizationTransformerV3(nn.Module):

    # Fused feature dim per AP: 52 (CSI) + 1 (RSSI) + 1 (SNR)
    FUSED_AP_DIM = 52 + 1 + 1  # 54

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_n1: int = 222,
        max_n2: int = 3317,
        max_k: int = 222,
        num_subcarriers: int = 52,
        max_building: int = 16,
        max_floor: int = 10,
        ffn_activation: str = "gelu",
        use_pre_norm: bool = True,
        num_pool_heads: int = 8,
        use_multi_head_pooling: bool = True,
        detach_logits_for_regression: bool = True,
        use_full_sequence_sin_pe: bool = False,
        max_seq_len: int = 1000,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_pre_norm = use_pre_norm
        self.max_n1 = max_n1
        self.max_n2 = max_n2
        self.max_k = max_k
        self.num_subcarriers = num_subcarriers
        self.max_building = max_building
        self.max_floor = max_floor
  
        self.detach_logits_for_regression = detach_logits_for_regression
 
        self.use_full_sequence_sin_pe = use_full_sequence_sin_pe
        self.full_seq_sin_pe: Optional[nn.Module]
        if use_full_sequence_sin_pe:
            self.full_seq_sin_pe = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        else:
            self.full_seq_sin_pe = None

        self.seq_len = 1 + max_n1

        self.ap2_encoder = nn.Linear(1, d_model)  

        self.global_pos = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.global_pos, std=0.02)

        # Per-AP fused projection: [B, N1, 54] → [B, N1, d_model] 
        self.fused_proj = nn.Sequential(
            nn.Linear(self.FUSED_AP_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Transformer encoder 
        if use_pre_norm:
            self.transformer = nn.ModuleList([
                PreNormEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=ffn_activation,
                )
                for _ in range(num_layers)
            ])
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation=ffn_activation,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer_norm = nn.LayerNorm(d_model)

        # Multi-Probe AP Aggregation (MPAA) 
        self.use_multi_head_pooling = use_multi_head_pooling
        if use_multi_head_pooling:
            self.num_pool_heads = num_pool_heads
            self.weight_proj = nn.Linear(d_model, num_pool_heads)  # W_pool [d → M]
            self.pool_fuse = nn.Sequential(
                nn.Linear(num_pool_heads * d_model, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.num_pool_heads = None
            self.weight_proj = None
            self.pool_fuse = None

        # Building head: h → bld_logits
        self.building_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, max_building),
        )

        # Floor head:
        self.floor_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, max_floor),
        )

        # GLO: logits embeddings 
        self.building_logits_embedding = nn.Sequential(
            nn.Linear(max_building, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.floor_logits_embedding = nn.Sequential(
            nn.Linear(max_floor, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # Regression head: [h || e_bld || e_flr] → coords
        self.reg_head = nn.Sequential(
            nn.Linear(3 * d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 2),
        )

    def forward(
        self,
        apvec_1: torch.Tensor,        # [B, N1]        AP presence flags
        apvec_2: torch.Tensor,        # [B, N2]        global AP signals
        csi_magnitude: torch.Tensor,  # [B, 52, K]     CSI per subcarrier per AP
        rssi: torch.Tensor,           # [B, N1]        RSSI per AP
        snr: torch.Tensor,            # [B, N1]        SNR per AP
        building_id: torch.Tensor,    # [B]            supervision target only
        floor_id: torch.Tensor,       # [B]            supervision target only
    ) -> Dict[str, torch.Tensor]:

        B  = apvec_1.size(0)
        N1 = apvec_1.size(1)


        K = csi_magnitude.size(2)
        assert K == N1, (
            f"CSI last dim K={K} must equal number of local APs N1={N1}. "
            f"Check data pipeline — do not silently truncate."
        )

        ap_pad_mask   = (apvec_1 == 0)                             # [B, N1]
        global_mask   = torch.zeros(B, 1, dtype=torch.bool,        
                                    device=apvec_1.device)
        key_padding_mask = torch.cat([global_mask, ap_pad_mask], dim=1)  # [B, 1+N1]

        # apvec_2 [B, N2] → [B, N2, 1] → Linear(1→d) → ReLU → [B, N2, d]
        e = F.relu(self.ap2_encoder(apvec_2.unsqueeze(-1)))        # [B, N2, d_model]
        global_vec   = e.mean(dim=1)                               # [B, d_model]

        global_token = (global_vec.unsqueeze(1) + self.global_pos) # [B, 1, d_model]

        # Feature fusion per AP: cat(csi, rssi, snr) 
        csi_per_ap = csi_magnitude.permute(0, 2, 1)               # [B, N1, 52]
        rssi_1     = rssi.unsqueeze(-1)                            # [B, N1, 1]
        snr_1      = snr.unsqueeze(-1)                             # [B, N1, 1]
        fused_ap   = torch.cat([csi_per_ap, rssi_1, snr_1], dim=-1)  # [B, N1, 54]

        # Gating 

        gate      = apvec_1.float().unsqueeze(-1)                 # [B, N1, 1]
        ap_tokens = self.fused_proj(fused_ap) * gate              # [B, N1, d_model]

        seq = torch.cat([global_token, ap_tokens], dim=1)         # [B, 1+N1, d_model]
        if self.full_seq_sin_pe is not None:
            seq = self.full_seq_sin_pe(seq)

        # Transformer encoder with key_padding_mask 
        if self.use_pre_norm:
            x = seq
            for layer in self.transformer:
                x = layer(x, src_key_padding_mask=key_padding_mask)
            out = x
        else:
            out = self.transformer(seq, src_key_padding_mask=key_padding_mask)
        out = self.transformer_norm(out)                           # [B, 1+N1, d_model]

 
        out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        # MPAA: multi-probe attention pooling → h 
        if self.use_multi_head_pooling:
            logits = self.weight_proj(out)
            logits = logits.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf"))
            weights = torch.softmax(logits, dim=1)                 # [B, L, M]
            head_summaries = torch.einsum('bld,blm->bdm', out, weights)   # [B, d, M]
            head_summaries = head_summaries.permute(0, 2, 1)       # [B, M, d_model]
            h = self.pool_fuse(head_summaries.reshape(B, -1))      # [B, d_model]
        else:
            valid = (~key_padding_mask).float().unsqueeze(-1)    # [B, L, 1]
            h = (out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        # ── Building head 
        building_logits = self.building_head(h)                    # [B, max_building]
        floor_logits = self.floor_head(h)                        # [B, max_floor]

        bld_in = building_logits.detach() if self.detach_logits_for_regression else building_logits
        flr_in = floor_logits.detach() if self.detach_logits_for_regression else floor_logits
        bld_emb = self.building_logits_embedding(bld_in)
        flr_emb = self.floor_logits_embedding(flr_in)

        # Regression head: cat(h, e_bld, e_flr) → coords 
        reg_input = torch.cat([h, bld_emb, flr_emb], dim=1)       # [B, 3*d_model]
        coords    = self.reg_head(reg_input)                       # [B, 2]

        return {
            "coords":          coords,
            "building_logits": building_logits,
            "floor_logits":    floor_logits,
        }


LocalizationTransformer = LocalizationTransformerV3
