"""Training script for LocalizationTransformer.
"""

import argparse
import copy
import pickle
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
from pathlib import Path
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from models.localization_transformer import LocalizationTransformer
from models.localization_transformer import LocalizationLoss
from data.localization_dataset import LocalizationDataset


def _compute_inverse_freq_ce_weights(
    train_pickle: Path,
    max_building: int,
    max_floor: int,
    power: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CE class weights from training.pkl counts; power=0.5 ~ sqrt inv-freq, 1.0 ~ full inv-freq."""
    with open(train_pickle, 'rb') as f:
        raw = pickle.load(f)
    y = np.asarray(raw['labels'], dtype=np.float64)
    b = np.clip(np.rint(y[:, 2]).astype(np.int64), 0, max_building - 1)
    fl = np.clip(np.rint(y[:, 3]).astype(np.int64), 0, max_floor - 1)

    def _w(yi: np.ndarray, ncls: int, p: float) -> torch.Tensor:
        cnt = np.bincount(yi, minlength=ncls).astype(np.float64)
        cnt = np.maximum(cnt, 1.0)
        if p <= 0:
            w = np.ones(ncls, dtype=np.float32)
        else:
            n = float(len(yi))
            k = float(ncls)
            w = (n / (k * cnt)) ** p
            w = w.astype(np.float32)
            w = w / w.mean()
        return torch.tensor(w)

    return _w(b, max_building, power), _w(fl, max_floor, power)


class LocalizationLightningModule(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.current_epoch_num = 0
        self.best_val_loss = float('inf')
        self.best_val_metrics = {}
        self.log_file = None

        m = config['model']
        self.model = LocalizationTransformer(
            d_model=m['d_model'],
            nhead=m['nhead'],
            num_layers=m['num_layers'],
            dim_feedforward=m['dim_feedforward'],
            dropout=m['dropout'],
            max_n1=m['max_n1'],
            max_n2=m['max_n2'],
            max_k=m['max_k'],
            num_subcarriers=m['num_subcarriers'],
            max_building=m['max_building'],
            max_floor=m['max_floor'],
            ffn_activation=m.get('ffn_activation', 'gelu'),
            use_pre_norm=m.get('use_pre_norm', True),
            num_pool_heads=m.get('num_pool_heads', 8),
            use_multi_head_pooling=m.get('use_multi_head_pooling', True),
            detach_logits_for_regression=m.get('detach_logits_for_regression', True),
            use_full_sequence_sin_pe=m.get('use_full_sequence_sin_pe', False),
            max_seq_len=m.get('max_seq_len', 1000),
        )
        t = config['training']
        b_ce_w, f_ce_w = None, None
        rew = str(t.get('cls_reweight', 'none')).lower()
        if rew in ('sqrt', 'inv'):
            p = 0.5 if rew == 'sqrt' else 1.0
            td = Path(config['data']['data_dir'])
            if not td.is_absolute():
                td = Path(__file__).resolve().parent / td
            tp = td / 'training.pkl'
            if tp.is_file():
                b_ce_w, f_ce_w = _compute_inverse_freq_ce_weights(
                    tp, m['max_building'], m['max_floor'], p,
                )
                print(f"CE class reweight ({rew}, p={p}) from {tp}")
            else:
                print(f"cls_reweight={rew} but missing {tp}; using uniform CE weights")
        self.criterion = LocalizationLoss(
            reg_weight=t['reg_weight'],
            building_weight=t['building_weight'],
            floor_weight=t['floor_weight'],
            max_building=m['max_building'],
            max_floor=m['max_floor'],
            huber_threshold=t.get('huber_threshold', 1.0),
            label_smoothing=t.get('label_smoothing', 0.0),
            building_class_weights=b_ce_w,
            floor_class_weights=f_ce_w,
        )
        # ReduceLROnPlateau (after warmup) steps on this metric; align with checkpoint when possible.
        self.plateau_monitor = t.get('plateau_monitor', t.get('monitor', 'val/total_loss'))
        self.plateau_mode = t.get('plateau_mode', t.get('monitor_mode', 'min'))

    def set_loss_weights(
        self,
        reg_weight: Optional[float] = None,
        building_weight: Optional[float] = None,
        floor_weight: Optional[float] = None,
    ) -> None:
        """Mutate criterion and config so a new stage can change loss balance without rebuilding the module."""
        if reg_weight is not None:
            self.criterion.reg_weight = float(reg_weight)
            self.config['training']['reg_weight'] = float(reg_weight)
        if building_weight is not None:
            self.criterion.building_weight = float(building_weight)
            self.config['training']['building_weight'] = float(building_weight)
        if floor_weight is not None:
            self.criterion.floor_weight = float(floor_weight)
            self.config['training']['floor_weight'] = float(floor_weight)

    def set_log_file(self, log_file_path):
        self.log_file = log_file_path

    def freeze_classifier_heads(self) -> None:
        """Stop training updates to building/floor logits heads (e.g. regression-only stage)."""
        for p in self.model.building_head.parameters():
            p.requires_grad = False
        for p in self.model.floor_head.parameters():
            p.requires_grad = False

    def forward(self, batch):
        # building_id and floor_id are passed to the model for the loss only;

        return self.model(
            apvec_1=batch['apvec_1'],
            apvec_2=batch['apvec_2'],
            csi_magnitude=batch['csi_magnitude'],
            rssi=batch['rssi'],
            snr=batch['snr'],
            building_id=batch['building_id'],
            floor_id=batch['floor_id'],
        )

    def training_step(self, batch, batch_idx):
        predictions = self.forward(batch)
        losses = self.criterion(predictions, batch['label'])
        coords_pred   = predictions['coords']
        coords_target = batch['label'][:, :2]
        reg_error     = torch.sqrt(torch.mean((coords_pred - coords_target) ** 2))
        building_pred = torch.argmax(predictions['building_logits'], dim=1)
        building_acc  = (building_pred == batch['building_id']).float().mean()
        floor_pred    = torch.argmax(predictions['floor_logits'], dim=1)
        floor_acc     = (floor_pred == batch['floor_id']).float().mean()
        mean_cls_acc  = 0.5 * (building_acc + floor_acc)
        if batch_idx % 10 == 0:
            print(
                f"Epoch {self.current_epoch_num:3d} | Train | Step {batch_idx:4d} | "
                f"Total: {losses['total_loss'].item():.4f} | "
                f"Reg: {losses['reg_loss'].item():.4f} | "
                f"RMSE: {reg_error.item():.4f} | "
                f"Bldg: {building_acc.item():.3f} | "
                f"Floor: {floor_acc.item():.3f}"
            )
            sys.stdout.flush()
        self.log('train/total_loss',   losses['total_loss'], on_step=True, on_epoch=True)
        self.log('train/reg_loss',     losses['reg_loss'],   on_step=True, on_epoch=True)
        self.log('train/reg_error',    reg_error,            on_step=True, on_epoch=True)
        self.log('train/building_acc', building_acc,         on_step=True, on_epoch=True)
        self.log('train/floor_acc',    floor_acc,            on_step=True, on_epoch=True)
        self.log('train/mean_cls_acc', mean_cls_acc,         on_step=True, on_epoch=True)
        return losses['total_loss']

    def validation_step(self, batch, batch_idx):
        predictions = self.forward(batch)
        losses = self.criterion(predictions, batch['label'])
        coords_pred   = predictions['coords']
        coords_target = batch['label'][:, :2]
        reg_error     = torch.sqrt(torch.mean((coords_pred - coords_target) ** 2))
        building_pred = torch.argmax(predictions['building_logits'], dim=1)
        building_acc  = (building_pred == batch['building_id']).float().mean()
        floor_pred    = torch.argmax(predictions['floor_logits'], dim=1)
        floor_acc     = (floor_pred == batch['floor_id']).float().mean()
        mean_cls_acc  = 0.5 * (building_acc + floor_acc)
        if batch_idx % 5 == 0:
            print(
                f"Epoch {self.current_epoch_num:3d} | Val   | Step {batch_idx:4d} | "
                f"Total: {losses['total_loss'].item():.4f} | "
                f"RMSE: {reg_error.item():.4f} | "
                f"Bldg: {building_acc.item():.3f} | "
                f"Floor: {floor_acc.item():.3f}"
            )
            sys.stdout.flush()
        self.log('val/total_loss',   losses['total_loss'],   on_step=False, on_epoch=True)
        self.log('val/reg_loss',     losses['reg_loss'],     on_step=False, on_epoch=True)
        self.log('val/building_loss', losses['building_loss'], on_step=False, on_epoch=True)
        self.log('val/floor_loss',    losses['floor_loss'],   on_step=False, on_epoch=True)
        self.log('val/reg_error',     reg_error,              on_step=False, on_epoch=True)
        self.log('val/building_acc',  building_acc,           on_step=False, on_epoch=True)
        self.log('val/floor_acc',     floor_acc,              on_step=False, on_epoch=True)
        self.log('val/mean_cls_acc',  mean_cls_acc,           on_step=False, on_epoch=True)
        return losses['total_loss']

    def test_step(self, batch, batch_idx):
        predictions = self.forward(batch)
        losses = self.criterion(predictions, batch['label'])
        coords_pred   = predictions['coords']
        coords_target = batch['label'][:, :2]
        reg_error     = torch.sqrt(torch.mean((coords_pred - coords_target) ** 2))
        building_pred = torch.argmax(predictions['building_logits'], dim=1)
        building_acc  = (building_pred == batch['building_id']).float().mean()
        floor_pred    = torch.argmax(predictions['floor_logits'], dim=1)
        floor_acc     = (floor_pred == batch['floor_id']).float().mean()
        self.log('test/total_loss',   losses['total_loss'], on_step=False, on_epoch=True)
        self.log('test/reg_error',    reg_error,            on_step=False, on_epoch=True)
        self.log('test/building_acc', building_acc,         on_step=False, on_epoch=True)
        self.log('test/floor_acc',    floor_acc,            on_step=False, on_epoch=True)
        return losses['total_loss']

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError('No trainable parameters; check freeze_* settings.')
        optimizer = torch.optim.Adam(
            params,
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
        )
        min_lr      = float(self.config['training']['min_lr'])
        lr_factor   = self.config['training'].get('lr_reduce_factor', 0.5)
        lr_patience = self.config['training'].get('lr_patience', 15)
        warmup_epochs = self.config['training'].get('warmup_epochs', 0)
        if warmup_epochs > 0:
            def lr_lambda(epoch):
                return (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            self.warmup_epochs = warmup_epochs
            self.main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=self.plateau_mode, factor=lr_factor,
                patience=lr_patience, min_lr=min_lr,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': warmup_scheduler, 'interval': 'epoch'},
            }
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=self.plateau_mode, factor=lr_factor,
            patience=lr_patience, min_lr=min_lr,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'monitor': self.plateau_monitor,
            },
        }

    def on_train_epoch_start(self):
        self.current_epoch_num = self.current_epoch

    def on_validation_epoch_end(self):
        if hasattr(self, 'warmup_epochs') and hasattr(self, 'main_scheduler'):
            current_epoch = getattr(self, 'current_epoch', 0)
            if current_epoch >= self.warmup_epochs:
                callback_metrics = getattr(self.trainer, 'callback_metrics', {})
                logged = getattr(self.trainer, 'logged_metrics', {})
                key = self.plateau_monitor
                val_m = callback_metrics.get(key) or logged.get(key)
                if val_m is not None:
                    self.main_scheduler.step(
                        val_m.item() if hasattr(val_m, 'item') else val_m
                    )
        try:
            callback_metrics = getattr(self.trainer, 'callback_metrics', {})
            logged = getattr(self.trainer, 'logged_metrics', {})
            total_loss   = callback_metrics.get('val/total_loss')   or logged.get('val/total_loss')
            reg_error    = callback_metrics.get('val/reg_error')    or logged.get('val/reg_error')
            building_acc = callback_metrics.get('val/building_acc') or logged.get('val/building_acc')
            floor_acc    = callback_metrics.get('val/floor_acc')    or logged.get('val/floor_acc')
            if total_loss is not None:
                def _v(x): return x.item() if x is not None and hasattr(x, 'item') else (x or 0.0)
                total_loss_val   = _v(total_loss)
                reg_error_val    = _v(reg_error)
                building_acc_val = _v(building_acc)
                floor_acc_val    = _v(floor_acc)
                print("\n" + "=" * 80)
                print(f"Epoch {self.current_epoch} Validation Summary:")
                print("=" * 80)
                print(f"  Total Loss: {total_loss_val:.4f}  RMSE: {reg_error_val:.4f}")
                print(f"  Building Acc: {building_acc_val:.3f}  Floor Acc: {floor_acc_val:.3f}")
                print("=" * 80 + "\n")
                sys.stdout.flush()
                if total_loss_val < self.best_val_loss:
                    self.best_val_loss = total_loss_val
                    reg_loss      = callback_metrics.get('val/reg_loss')      or logged.get('val/reg_loss')
                    building_loss = callback_metrics.get('val/building_loss') or logged.get('val/building_loss')
                    floor_loss    = callback_metrics.get('val/floor_loss')    or logged.get('val/floor_loss')
                    self.best_val_metrics = {
                        'epoch':         self.current_epoch,
                        'total_loss':    total_loss_val,
                        'reg_loss':      _v(reg_loss),
                        'building_loss': _v(building_loss),
                        'floor_loss':    _v(floor_loss),
                        'rmse':          reg_error_val,
                        'building_acc':  building_acc_val,
                        'floor_acc':     floor_acc_val,
                    }
                    if self.log_file:
                        self.save_best_results_to_log()
        except Exception as e:
            print(f"\nWarning: Could not print validation summary: {e}\n")
            sys.stdout.flush()

    def save_best_results_to_log(self):
        if not self.log_file or not self.best_val_metrics:
            return
        try:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("BEST EVALUATION RESULTS\n")
                f.write("=" * 80 + "\n")
                f.write(f"Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model:      LocalizationTransformer\n")
                f.write(f"Best Epoch: {self.best_val_metrics['epoch']}\n")
                f.write(f"Best Val Loss: {self.best_val_metrics['total_loss']:.6f}\n")
                f.write(f"RMSE:          {self.best_val_metrics['rmse']:.6f}\n")
                f.write(f"Building Acc:  {self.best_val_metrics['building_acc']:.4f}\n")
                f.write(f"Floor Acc:     {self.best_val_metrics['floor_acc']:.4f}\n")
                f.write("=" * 80 + "\n")
        except Exception as e:
            print(f"Warning: Could not save log file: {e}")


def create_data_loaders(config, train_dir=None, eval_dir=None, test_dir=None):
    """Create train/val/test data loaders."""
    data_dir = Path(config['data']['data_dir'])
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent / data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    def resolve(p):
        pp = Path(p)
        if not pp.is_absolute():
            pp = Path(__file__).parent / pp
        return pp

    train_base = resolve(train_dir) if train_dir else data_dir
    eval_base  = resolve(eval_dir)  if eval_dir  else data_dir
    test_base  = resolve(test_dir)  if test_dir  else data_dir
    train_file = train_base / 'training.pkl'
    val_file   = eval_base  / 'eval.pkl'
    test_file  = test_base  / 'test.pkl'
    for f, name in [(train_file, 'Training'), (val_file, 'Eval'), (test_file, 'Test')]:
        if not f.exists():
            raise FileNotFoundError(f"{name} file not found: {f}")
    print(f"Loading data: train={train_file.parent}, eval={val_file.parent}, test={test_file.parent}")
    num_workers = 0 if sys.platform == 'win32' else config['data']['num_workers']
    train_loader = DataLoader(
        LocalizationDataset(str(train_file)),
        batch_size=config['data']['batch_size'], shuffle=True,
        num_workers=num_workers, pin_memory=(num_workers > 0),
    )
    val_loader = DataLoader(
        LocalizationDataset(str(val_file)),
        batch_size=config['data']['batch_size'], shuffle=False,
        num_workers=num_workers, pin_memory=(num_workers > 0),
    )
    test_loader = DataLoader(
        LocalizationDataset(str(test_file)),
        batch_size=config['data']['batch_size'], shuffle=False,
        num_workers=num_workers, pin_memory=(num_workers > 0),
    )
    return train_loader, val_loader, test_loader


_STAGE_TRAINER_KEYS = frozenset({'max_epochs', 'patience'})
# These are merged into training.* (Lightning reads them); not Trainer() kwargs.
# monitor / monitor_mode: checkpoint + early stopping + default plateau behavior.
# plateau_* optional overrides for LR scheduler only.
# Not merged into training.* — handled explicitly when building the Lightning module / stage.
_STAGE_PROCEDURE_KEYS = frozenset({
    'freeze_classifier_heads',
    'detach_logits_for_regression',
})


def _split_stage_section(stage: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Split YAML stage block into training overrides, Trainer kwargs, and procedure flags."""
    train_ov: Dict[str, Any] = {}
    trainer_kw: Dict[str, Any] = {}
    procedure: Dict[str, Any] = {}
    for k, v in stage.items():
        if k in _STAGE_TRAINER_KEYS:
            trainer_kw[k] = v
        elif k in _STAGE_PROCEDURE_KEYS:
            procedure[k] = v
        else:
            train_ov[k] = v
    return train_ov, trainer_kw, procedure


def _apply_procedure_to_config_model(cfg: dict, procedure: Dict[str, Any]) -> None:
    if 'detach_logits_for_regression' in procedure:
        cfg.setdefault('model', {})['detach_logits_for_regression'] = bool(
            procedure['detach_logits_for_regression']
        )


def _merge_training(base_training: dict, overrides: dict) -> dict:
    return {**base_training, **overrides}


def _apply_regression_best_checkpoint_monitor(cfg: dict) -> None:
    """Use val/reg_loss for ModelCheckpoint, EarlyStopping, and LR plateau (unless plateau_* set in YAML).

    Stage 1 (classification pretrain) does not call this; stage 2+, single-stage, and stage 3 do.
    """
    t = cfg['training']
    t['monitor'] = 'val/reg_loss'
    t['monitor_mode'] = 'min'
    if 'plateau_monitor' not in t:
        t['plateau_monitor'] = 'val/reg_loss'
    if 'plateau_mode' not in t:
        t['plateau_mode'] = 'min'


def _resolve_accelerator(config: dict) -> str:
    accelerator = config['system']['accelerator']
    if accelerator == 'gpu' and not torch.cuda.is_available():
        print('CUDA not available; using CPU.')
        accelerator = 'cpu'
    return accelerator


def _run_stage_fit(
    model: LocalizationLightningModule,
    config: dict,
    train_loader,
    val_loader,
    *,
    stage_label: str,
    checkpoint_dir: Path,
    max_epochs: int,
    patience: int,
    accelerator: str,
    tb_name: str,
) -> Tuple[ModelCheckpoint, pl.Trainer]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    t = config['training']
    ckpt_monitor = t.get('monitor', 'val/total_loss')
    ckpt_mode = t.get('monitor_mode', 'min')
    # Metric names in filenames break on '/'; keep epoch-only name when monitor is not total_loss.
    if ckpt_monitor == 'val/total_loss' and ckpt_mode == 'min':
        ckpt_fn = f'{stage_label}-best-{{epoch:02d}}-val_loss={{val/total_loss:.4f}}'
    elif ckpt_monitor == 'val/reg_loss' and ckpt_mode == 'min':
        ckpt_fn = f'{stage_label}-best-{{epoch:02d}}-reg_loss={{val/reg_loss:.4f}}'
    else:
        ckpt_fn = f'{stage_label}-best-{{epoch:02d}}'
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        monitor=ckpt_monitor,
        mode=ckpt_mode,
        save_top_k=1,
        filename=ckpt_fn,
        auto_insert_metric_name=False,
        save_last=True,
        verbose=True,
    )
    early_stop_callback = EarlyStopping(
        monitor=ckpt_monitor,
        mode=ckpt_mode,
        patience=patience,
        verbose=True,
    )
    logger = TensorBoardLogger(
        save_dir=config['logging']['log_dir'],
        name=tb_name,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=config['system']['devices'] if accelerator != 'cpu' else 1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        precision=config['training']['precision'],
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
    )
    print(
        f"\n{'=' * 60}\n"
        f"  {stage_label}:  reg_weight={t['reg_weight']}  "
        f"building_weight={t['building_weight']}  floor_weight={t['floor_weight']}\n"
        f"  max_epochs={max_epochs}  patience={patience}  "
        f"lr={t['learning_rate']}  ckpt/early_stop: {ckpt_monitor} ({ckpt_mode})\n"
        f"  plateau: {t.get('plateau_monitor', t.get('monitor', 'val/total_loss'))} "
        f"({t.get('plateau_mode', t.get('monitor_mode', 'min'))})\n"
        f"{'=' * 60}\n"
    )
    sys.stdout.flush()
    trainer.fit(model, train_loader, val_loader)
    return checkpoint_callback, trainer


def _post_train_save_and_test(
    model_for_log: LocalizationLightningModule,
    model_checkpoint_cb: ModelCheckpoint,
    config: dict,
    test_loader,
    trainer: pl.Trainer,
    logs_dir: Path,
    state_dict_name: str,
) -> LocalizationLightningModule:
    map_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model = None
    if model_checkpoint_cb.best_model_path:
        best_model = LocalizationLightningModule.load_from_checkpoint(
            model_checkpoint_cb.best_model_path,
            config=config,
            map_location=map_device,
        )
        best_model.eval()
        torch.save(best_model.model.state_dict(), logs_dir / state_dict_name)
        print(f"Best backbone saved to: {logs_dir / state_dict_name}")

    last_model = best_model if best_model is not None else model_for_log
    last_model.eval()
    torch.save(last_model.model.state_dict(), logs_dir / 'last_epoch_model.pt')
    trainer.test(last_model, test_loader)
    return last_model


def _interpret_stage_args(stage_args: list) -> Optional[int]:
    """Return 1, 2, or 3 for `stage 1` / `stage 2` / `stage 3` CLI, else None."""
    if not stage_args:
        return None
    joined = ' '.join(str(x) for x in stage_args).strip().lower()
    toks = joined.split()
    if toks in (['stage', '1'], ['1']) or joined == 'stage1':
        return 1
    if toks in (['stage', '2'], ['2']) or joined == 'stage2':
        return 2
    if toks in (['stage', '3'], ['3']) or joined == 'stage3':
        return 3
    raise SystemExit(
        f'Unrecognized arguments {stage_args!r}. '
        f'Use "python main.py stage 1" | "stage 2" | "stage 3", or omit for default behavior.'
    )


def _cfg_for_checkpoint_eval(config: dict, ckpt_path: Path) -> dict:
    """Match training hyperparameters to the checkpoint (stage1 / stage2 / stage3 vs single)."""
    cfg = copy.deepcopy(config)
    two = cfg.get('two_stage') or {}
    path_s = str(ckpt_path).lower().replace('\\', '/')
    if 'stage1' in path_s or 's1_cls' in path_s or 'stage1_pretrained' in path_s:
        if 'stage1' in two:
            ov, tr, proc = _split_stage_section(copy.deepcopy(two['stage1']))
            _apply_procedure_to_config_model(cfg, proc)
            cfg['training'] = _merge_training(cfg['training'], ov)
    elif 'stage3' in path_s or 's3_ft' in path_s:
        if 'stage3' in two:
            ov, tr, proc = _split_stage_section(copy.deepcopy(two['stage3']))
            _apply_procedure_to_config_model(cfg, proc)
            cfg['training'] = _merge_training(cfg['training'], ov)
    elif 'stage2' in path_s or 's2_reg' in path_s or 'stage2_pretrained' in path_s:
        if 'stage2' in two:
            ov, tr, proc = _split_stage_section(copy.deepcopy(two['stage2']))
            _apply_procedure_to_config_model(cfg, proc)
            cfg['training'] = _merge_training(cfg['training'], ov)
    elif two.get('use_three_stage') and 'stage3' in two:
        ov, tr, proc = _split_stage_section(copy.deepcopy(two['stage3']))
        _apply_procedure_to_config_model(cfg, proc)
        cfg['training'] = _merge_training(cfg['training'], ov)
    elif 'stage2' in two:
        ov, tr, proc = _split_stage_section(copy.deepcopy(two['stage2']))
        _apply_procedure_to_config_model(cfg, proc)
        cfg['training'] = _merge_training(cfg['training'], ov)
    return cfg


def _resolve_test_checkpoint(ckpt_arg: Optional[Path], ck_root: Path) -> Path:
    if ckpt_arg is not None:
        p = Path(ckpt_arg)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / p
        if not p.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {p}')
        return p
    search_dirs = [
        ck_root / 'stage3_ft',
        ck_root / 'stage2_reg',
        ck_root / 'stage1_cls',
        ck_root,
    ]
    for d in search_dirs:
        if not d.is_dir():
            continue
        cks = sorted(d.glob('*.ckpt'), key=lambda x: x.stat().st_mtime, reverse=True)
        if cks:
            print(f'Using newest checkpoint: {cks[0]}')
            return cks[0]
    raise FileNotFoundError(
        f'No .ckpt found under {ck_root} (stage2_reg / stage1_cls / root). '
        f'Pass --ckpt path/to/model.ckpt'
    )


def _resolve_repo_path(p: Path) -> Path:
    pp = Path(p)
    if not pp.is_absolute():
        pp = Path(__file__).resolve().parent / pp
    return pp


def _run_test_only(
    config: dict,
    ckpt_path: Path,
    test_loader,
    accelerator: str,
    log_dir: Path,
) -> None:
    eval_cfg = _cfg_for_checkpoint_eval(config, ckpt_path)
    map_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LocalizationLightningModule.load_from_checkpoint(
        str(ckpt_path),
        config=eval_cfg,
        map_location=map_device,
    )
    model.eval()
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=config['system']['devices'] if accelerator != 'cpu' else 1,
        logger=TensorBoardLogger(save_dir=str(log_dir), name='test_eval'),
    )
    print('Running test on test.pkl ...')
    trainer.test(model, test_loader)


def _run_test_only_weights(
    config: dict,
    weights_path: Path,
    test_loader,
    accelerator: str,
    log_dir: Path,
) -> None:
    """Load backbone from torch.save (e.g. results/logs/best_model.pt)."""
    eval_cfg = _cfg_for_checkpoint_eval(config, weights_path)
    map_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LocalizationLightningModule(eval_cfg)
    state = torch.load(str(weights_path), map_location=map_device)
    model.model.load_state_dict(state, strict=True)
    model.eval()
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=config['system']['devices'] if accelerator != 'cpu' else 1,
        logger=TensorBoardLogger(save_dir=str(log_dir), name='test_eval'),
    )
    print(f'Loaded weights from {weights_path}')
    print('Running test on test.pkl ...')
    trainer.test(model, test_loader)


def _find_stage1_checkpoint(ck_root: Path) -> Optional[Path]:
    """Stable copy from a prior stage-1 run, else newest ckpt under stage1_cls/."""
    alias = ck_root / 'stage1_pretrained.ckpt'
    if alias.is_file():
        return alias
    d = ck_root / 'stage1_cls'
    if not d.is_dir():
        return None
    cks = sorted(d.glob('*.ckpt'), key=lambda p: p.stat().st_mtime, reverse=True)
    return cks[0] if cks else None


def _find_stage2_checkpoint(ck_root: Path) -> Optional[Path]:
    """Stable copy from a prior stage-2 run, else newest ckpt under stage2_reg/."""
    alias = ck_root / 'stage2_pretrained.ckpt'
    if alias.is_file():
        return alias
    d = ck_root / 'stage2_reg'
    if not d.is_dir():
        return None
    cks = sorted(d.glob('*.ckpt'), key=lambda p: p.stat().st_mtime, reverse=True)
    return cks[0] if cks else None


def _parse_args():
    p = argparse.ArgumentParser(
        description='Train LocalizationTransformer.',
        epilog='Examples:  python main.py stage 1   # classification only (YAML stage1)',
    )
    p.add_argument(
        '--config',
        type=Path,
        default=Path('cfg/localization_config.yaml'),
        help='Path to YAML config (default: cfg/localization_config.yaml)',
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        '--single-stage',
        action='store_true',
        help='Ignore two_stage.enabled in YAML and run one training phase.',
    )
    g.add_argument(
        '--two-stage',
        action='store_true',
        help='Force two-stage training even if two_stage.enabled is false.',
    )
    p.add_argument(
        'stage_args',
        nargs='*',
        metavar='STAGE',
        help='Optional: stage 1 / 2 / 3 (see two_stage in YAML).',
    )
    p.add_argument(
        '--test-only',
        action='store_true',
        help='Skip training; run test on data/test.pkl (Lightning .ckpt or backbone .pt).',
    )
    p.add_argument(
        '--ckpt',
        type=Path,
        default=None,
        help='Lightning checkpoint .ckpt for --test-only (see also --weights).',
    )
    p.add_argument(
        '--weights',
        type=Path,
        default=None,
        help='Backbone state dict .pt (e.g. results/logs/best_model.pt). Overrides --ckpt search.',
    )
    return p.parse_args()


def main():
    args = _parse_args()
    only_stage = _interpret_stage_args(args.stage_args)
    if args.test_only and only_stage is not None:
        raise SystemExit('Do not combine --test-only with "stage 1" / "stage 2" / "stage 3" positional args.')
    if only_stage is not None and args.single_stage:
        raise SystemExit('Cannot use --single-stage together with "stage 1" / "stage 2" / "stage 3".')
    config_path = args.config
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    repo_root = Path(__file__).resolve().parent
    if not Path(base_config['logging']['log_dir']).is_absolute():
        log_dir = repo_root / base_config['logging']['log_dir']
    else:
        log_dir = Path(base_config['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)

    config = copy.deepcopy(base_config)
    config['logging'] = config.get('logging', {}).copy()
    config['logging']['experiment_name'] = 'localization'
    config['logging']['log_dir'] = str(log_dir)

    seed = int(config.get('system', {}).get('seed', 42))
    try:
        pl.seed_everything(seed, workers=True)
    except TypeError:
        pl.seed_everything(seed)

    train_loader, val_loader, test_loader = create_data_loaders(config)
    accelerator = _resolve_accelerator(config)

    logs_dir = log_dir
    log_file = logs_dir / 'best_eval_results.txt'

    exp_name = config['logging']['experiment_name']
    ck_root = log_dir / exp_name / 'checkpoints'

    if args.test_only:
        if args.weights is not None:
            wpath = _resolve_repo_path(args.weights)
            if not wpath.is_file():
                raise FileNotFoundError(f'Weights file not found: {wpath}')
            _run_test_only_weights(config, wpath, test_loader, accelerator, log_dir)
            return
        if args.ckpt is not None:
            p = _resolve_repo_path(args.ckpt)
            if not p.is_file():
                raise FileNotFoundError(f'Checkpoint not found: {p}')
            if p.suffix.lower() == '.pt':
                _run_test_only_weights(config, p, test_loader, accelerator, log_dir)
            else:
                _run_test_only(config, p, test_loader, accelerator, log_dir)
            return
        try:
            ckpt_path = _resolve_test_checkpoint(None, ck_root)
            _run_test_only(config, ckpt_path, test_loader, accelerator, log_dir)
        except FileNotFoundError:
            default_pt = logs_dir / 'best_model.pt'
            if default_pt.is_file():
                print(f'No .ckpt found; using backbone weights: {default_pt}')
                _run_test_only_weights(config, default_pt, test_loader, accelerator, log_dir)
            else:
                raise FileNotFoundError(
                    f'No Lightning .ckpt under {ck_root} and no {default_pt}. '
                    f'Pass --weights results/logs/best_model.pt or --ckpt path/to/model.ckpt'
                ) from None
        return

    two = config.get('two_stage') or {}
    use_two_stage = bool(two.get('enabled'))
    if args.two_stage:
        use_two_stage = True
    elif args.single_stage:
        use_two_stage = False

    # ----- stage 1 only (classification pretrain) -----
    if only_stage == 1:
        if 'stage1' not in two:
            raise ValueError('CLI "stage 1" requires two_stage.stage1 in the config YAML.')
        ov1, tr1, proc1 = _split_stage_section(copy.deepcopy(two['stage1']))
        cfg1 = copy.deepcopy(config)
        _apply_procedure_to_config_model(cfg1, proc1)
        cfg1['training'] = _merge_training(config['training'], ov1)
        max_e1 = int(tr1.get('max_epochs', cfg1['training']['max_epochs']))
        pat1 = int(tr1.get('patience', cfg1['training']['patience']))

        model1 = LocalizationLightningModule(cfg1)
        model1.set_log_file(str(log_file))
        cb1, trainer1 = _run_stage_fit(
            model1,
            cfg1,
            train_loader,
            val_loader,
            stage_label='s1_cls',
            checkpoint_dir=ck_root / 'stage1_cls',
            max_epochs=max_e1,
            patience=pat1,
            accelerator=accelerator,
            tb_name=f'{exp_name}_stage1_cls',
        )
        ckpt1_path = cb1.best_model_path or cb1.last_model_path
        if not ckpt1_path:
            raise RuntimeError('Stage 1 finished without a checkpoint path')
        print(f"Stage 1 best/lightning ckpt: {ckpt1_path}")
        stage1_alias = ck_root / 'stage1_pretrained.ckpt'
        try:
            shutil.copy2(ckpt1_path, stage1_alias)
            print(f"Copied to stable path: {stage1_alias}")
        except OSError as e:
            print(f"Note: could not copy stage-1 ckpt to {stage1_alias}: {e}")

        _post_train_save_and_test(
            model1,
            cb1,
            cfg1,
            test_loader,
            trainer1,
            logs_dir,
            'best_model_stage1.pt',
        )
        model1.save_best_results_to_log()
        print(f"\nStage 1 only: eval results saved to: {log_file}")
        return

    # ----- stage 2 only (regression; load stage-1 ckpt) -----
    if only_stage == 2:
        if 'stage2' not in two:
            raise ValueError('CLI "stage 2" requires two_stage.stage2 in the config YAML.')
        ckpt1_path = _find_stage1_checkpoint(ck_root)
        if not ckpt1_path or not ckpt1_path.is_file():
            raise FileNotFoundError(
                f'No stage-1 checkpoint found under {ck_root}. '
                f'Run "python main.py stage 1" first (or full two-stage run).'
            )
        print(f'Loading stage-1 weights from: {ckpt1_path}')

        ov2, tr2, proc2 = _split_stage_section(copy.deepcopy(two['stage2']))
        cfg2 = copy.deepcopy(config)
        _apply_procedure_to_config_model(cfg2, proc2)
        cfg2['training'] = _merge_training(config['training'], ov2)
        _apply_regression_best_checkpoint_monitor(cfg2)
        max_e2 = int(tr2.get('max_epochs', cfg2['training']['max_epochs']))
        pat2 = int(tr2.get('patience', cfg2['training']['patience']))

        map_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model2 = LocalizationLightningModule.load_from_checkpoint(
            str(ckpt1_path),
            config=cfg2,
            map_location=map_device,
        )
        model2.set_log_file(str(log_file))
        model2.best_val_loss = float('inf')
        model2.best_val_metrics = {}

        tw = cfg2['training']
        freeze_heads = proc2.get('freeze_classifier_heads')
        if freeze_heads is None:
            freeze_heads = tw['building_weight'] == 0 and tw['floor_weight'] == 0
        if freeze_heads:
            model2.freeze_classifier_heads()
            print('Stage 2: building_head & floor_head frozen (classification heads).')

        cb2, trainer2 = _run_stage_fit(
            model2,
            cfg2,
            train_loader,
            val_loader,
            stage_label='s2_reg',
            checkpoint_dir=ck_root / 'stage2_reg',
            max_epochs=max_e2,
            patience=pat2,
            accelerator=accelerator,
            tb_name=f'{exp_name}_stage2_reg',
        )

        last_model = _post_train_save_and_test(
            model2, cb2, cfg2, test_loader, trainer2, logs_dir, 'best_model.pt',
        )
        model2.save_best_results_to_log()
        ck2 = cb2.best_model_path or cb2.last_model_path
        if ck2:
            try:
                shutil.copy2(ck2, ck_root / 'stage2_pretrained.ckpt')
                print(f"Copied to stable path: {ck_root / 'stage2_pretrained.ckpt'}")
            except OSError as e:
                print(f"Note: could not copy stage-2 ckpt: {e}")
        print(f"\nStage 2 only: best eval results saved to: {log_file}")
        return

    # ----- stage 3 only (joint fine-tune from stage-2 ckpt) -----
    if only_stage == 3:
        if 'stage3' not in two:
            raise ValueError('CLI "stage 3" requires two_stage.stage3 in the config YAML.')
        ckpt2_path = _find_stage2_checkpoint(ck_root)
        if not ckpt2_path or not ckpt2_path.is_file():
            raise FileNotFoundError(
                f'No stage-2 checkpoint under {ck_root}. '
                f'Run stage 2 first (full pipeline or "python main.py stage 2").'
            )
        print(f'Loading stage-2 weights from: {ckpt2_path}')

        ov3, tr3, proc3 = _split_stage_section(copy.deepcopy(two['stage3']))
        cfg3 = copy.deepcopy(config)
        _apply_procedure_to_config_model(cfg3, proc3)
        cfg3['training'] = _merge_training(config['training'], ov3)
        _apply_regression_best_checkpoint_monitor(cfg3)
        max_e3 = int(tr3.get('max_epochs', cfg3['training']['max_epochs']))
        pat3 = int(tr3.get('patience', cfg3['training']['patience']))

        map_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model3 = LocalizationLightningModule.load_from_checkpoint(
            str(ckpt2_path),
            config=cfg3,
            map_location=map_device,
        )
        model3.set_log_file(str(log_file))
        model3.best_val_loss = float('inf')
        model3.best_val_metrics = {}

        fh3 = proc3.get('freeze_classifier_heads')
        if fh3 is None:
            fh3 = False
        if fh3:
            model3.freeze_classifier_heads()
            print('Stage 3: building_head & floor_head frozen.')
        else:
            print('Stage 3: joint fine-tune — regression + small CE.')

        cb3, trainer3 = _run_stage_fit(
            model3,
            cfg3,
            train_loader,
            val_loader,
            stage_label='s3_ft',
            checkpoint_dir=ck_root / 'stage3_ft',
            max_epochs=max_e3,
            patience=pat3,
            accelerator=accelerator,
            tb_name=f'{exp_name}_stage3_ft',
        )
        _post_train_save_and_test(
            model3, cb3, cfg3, test_loader, trainer3, logs_dir, 'best_model.pt',
        )
        model3.save_best_results_to_log()
        print(f"\nStage 3 only: best eval results saved to: {log_file}")
        return

    if use_two_stage:
        if 'stage1' not in two or 'stage2' not in two:
            raise ValueError("two_stage.enabled requires two_stage.stage1 and two_stage.stage2")
        if bool(two.get('use_three_stage')) and 'stage3' not in two:
            raise ValueError("two_stage.use_three_stage requires two_stage.stage3 in the YAML")

        ov1, tr1, proc1 = _split_stage_section(copy.deepcopy(two['stage1']))
        cfg1 = copy.deepcopy(config)
        _apply_procedure_to_config_model(cfg1, proc1)
        cfg1['training'] = _merge_training(config['training'], ov1)
        max_e1 = int(tr1.get('max_epochs', cfg1['training']['max_epochs']))
        pat1 = int(tr1.get('patience', cfg1['training']['patience']))

        model1 = LocalizationLightningModule(cfg1)
        model1.set_log_file(str(log_file))
        cb1, trainer1 = _run_stage_fit(
            model1,
            cfg1,
            train_loader,
            val_loader,
            stage_label='s1_cls',
            checkpoint_dir=ck_root / 'stage1_cls',
            max_epochs=max_e1,
            patience=pat1,
            accelerator=accelerator,
            tb_name=f'{exp_name}_stage1_cls',
        )
        ckpt1_path = cb1.best_model_path or cb1.last_model_path
        if not ckpt1_path:
            raise RuntimeError('Stage 1 finished without a checkpoint path')
        print(f"Stage 1 best/lightning ckpt: {ckpt1_path}")
        stage1_alias = ck_root / 'stage1_pretrained.ckpt'
        try:
            shutil.copy2(ckpt1_path, stage1_alias)
            print(f"Copied to stable path: {stage1_alias}")
        except OSError as e:
            print(f"Note: could not copy stage-1 ckpt to {stage1_alias}: {e}")

        ov2, tr2, proc2 = _split_stage_section(copy.deepcopy(two['stage2']))
        cfg2 = copy.deepcopy(config)
        _apply_procedure_to_config_model(cfg2, proc2)
        cfg2['training'] = _merge_training(config['training'], ov2)
        _apply_regression_best_checkpoint_monitor(cfg2)
        max_e2 = int(tr2.get('max_epochs', cfg2['training']['max_epochs']))
        pat2 = int(tr2.get('patience', cfg2['training']['patience']))

        map_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model2 = LocalizationLightningModule.load_from_checkpoint(
            ckpt1_path,
            config=cfg2,
            map_location=map_device,
        )
        model2.set_log_file(str(log_file))
        model2.best_val_loss = float('inf')
        model2.best_val_metrics = {}

        tw = cfg2['training']
        freeze_heads = proc2.get('freeze_classifier_heads')
        if freeze_heads is None:
            freeze_heads = tw['building_weight'] == 0 and tw['floor_weight'] == 0
        if freeze_heads:
            model2.freeze_classifier_heads()
            print('Stage 2: building_head & floor_head frozen (classification heads).')

        cb2, trainer2 = _run_stage_fit(
            model2,
            cfg2,
            train_loader,
            val_loader,
            stage_label='s2_reg',
            checkpoint_dir=ck_root / 'stage2_reg',
            max_epochs=max_e2,
            patience=pat2,
            accelerator=accelerator,
            tb_name=f'{exp_name}_stage2_reg',
        )

        ckpt2_path = cb2.best_model_path or cb2.last_model_path
        if not ckpt2_path:
            raise RuntimeError('Stage 2 finished without a checkpoint path')
        try:
            shutil.copy2(ckpt2_path, ck_root / 'stage2_pretrained.ckpt')
            print(f"Copied to stable path: {ck_root / 'stage2_pretrained.ckpt'}")
        except OSError as e:
            print(f"Note: could not copy stage-2 ckpt: {e}")

        use_three = bool(two.get('use_three_stage')) and 'stage3' in two
        if use_three:
            ov3, tr3, proc3 = _split_stage_section(copy.deepcopy(two['stage3']))
            cfg3 = copy.deepcopy(config)
            _apply_procedure_to_config_model(cfg3, proc3)
            cfg3['training'] = _merge_training(config['training'], ov3)
            _apply_regression_best_checkpoint_monitor(cfg3)
            max_e3 = int(tr3.get('max_epochs', cfg3['training']['max_epochs']))
            pat3 = int(tr3.get('patience', cfg3['training']['patience']))

            model3 = LocalizationLightningModule.load_from_checkpoint(
                str(ckpt2_path),
                config=cfg3,
                map_location=map_device,
            )
            model3.set_log_file(str(log_file))
            model3.best_val_loss = float('inf')
            model3.best_val_metrics = {}

            fh3 = proc3.get('freeze_classifier_heads')
            if fh3 is None:
                fh3 = False
            if fh3:
                model3.freeze_classifier_heads()
                print('Stage 3: building_head & floor_head frozen.')
            else:
                print('Stage 3: joint fine-tune — regression + small CE (heads trainable).')

            cb3, trainer3 = _run_stage_fit(
                model3,
                cfg3,
                train_loader,
                val_loader,
                stage_label='s3_ft',
                checkpoint_dir=ck_root / 'stage3_ft',
                max_epochs=max_e3,
                patience=pat3,
                accelerator=accelerator,
                tb_name=f'{exp_name}_stage3_ft',
            )
            _post_train_save_and_test(
                model3, cb3, cfg3, test_loader, trainer3, logs_dir, 'best_model.pt',
            )
            model3.save_best_results_to_log()
        else:
            _post_train_save_and_test(
                model2, cb2, cfg2, test_loader, trainer2, logs_dir, 'best_model.pt',
            )
            model2.save_best_results_to_log()
        print(f"\nBest eval results saved to: {log_file}")
        return

    # ----- single-stage (not stage-1 cls pretrain): best ckpt by val/reg_loss -----
    _apply_regression_best_checkpoint_monitor(config)
    model = LocalizationLightningModule(config)
    model.set_log_file(str(log_file))

    cb, trainer = _run_stage_fit(
        model,
        config,
        train_loader,
        val_loader,
        stage_label='single',
        checkpoint_dir=ck_root,
        max_epochs=int(config['training']['max_epochs']),
        patience=int(config['training']['patience']),
        accelerator=accelerator,
        tb_name=exp_name,
    )
    _post_train_save_and_test(
        model, cb, config, test_loader, trainer, logs_dir, 'best_model.pt',
    )
    model.save_best_results_to_log()
    print(f"\nBest eval results saved to: {log_file}")


if __name__ == '__main__':
    main()
