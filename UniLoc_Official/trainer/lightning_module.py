"""PyTorch Lightning module for transformer training."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from typing import Dict, Any, Optional

from models.transformer import VanillaTransformer


class TransformerLightningModule(pl.LightningModule):
    """PyTorch Lightning module for transformer training."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Lightning module.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Initialize model
        model_config = config['model']
        self.model = VanillaTransformer(
            vocab_size=model_config['vocab_size'],
            d_model=model_config['d_model'],
            nhead=model_config['nhead'],
            num_encoder_layers=model_config['num_encoder_layers'],
            num_decoder_layers=model_config['num_decoder_layers'],
            dim_feedforward=model_config['dim_feedforward'],
            dropout=model_config['dropout'],
            activation=model_config['activation'],
            max_seq_length=model_config['max_seq_length']
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is padding token
        
        # Training config
        self.training_config = config['training']
        self.optimizer_config = config.get('optimizer', {})
        self.scheduler_config = config.get('scheduler', {})
        
    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """Forward pass.
        
        Args:
            src: Source sequence
            tgt: Target sequence
            
        Returns:
            Model output
        """
        return self.model(src, tgt)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Training step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        src = batch['src'].transpose(0, 1)  # (seq_len, batch_size)
        tgt = batch['tgt'].transpose(0, 1)  # (seq_len, batch_size)
        
        # For teacher forcing: use tgt[:-1] as input and tgt[1:] as target
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]
        
        # Forward pass
        output = self.model(src, tgt_input)
        
        # Reshape for loss calculation
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        
        # Calculate loss
        loss = self.criterion(output, tgt_output)
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Validation loss
        """
        src = batch['src'].transpose(0, 1)
        tgt = batch['tgt'].transpose(0, 1)
        
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]
        
        output = self.model(src, tgt_input)
        
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        
        loss = self.criterion(output, tgt_output)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Test step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Test loss
        """
        src = batch['src'].transpose(0, 1)
        tgt = batch['tgt'].transpose(0, 1)
        
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]
        
        output = self.model(src, tgt_input)
        
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        
        loss = self.criterion(output, tgt_output)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler.
        
        Returns:
            Optimizer and scheduler configuration
        """
        # Get optimizer parameters
        lr = self.training_config['learning_rate']
        weight_decay = self.training_config.get('weight_decay', 0.0)
        optimizer_name = self.optimizer_config.get('name', 'adam').lower()
        
        # Create optimizer
        if optimizer_name == 'adamw':
            optimizer = AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=self.optimizer_config.get('betas', [0.9, 0.999]),
                eps=self.optimizer_config.get('eps', 1e-8)
            )
        else:  # Default to Adam
            optimizer = Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=self.optimizer_config.get('betas', [0.9, 0.999]),
                eps=self.optimizer_config.get('eps', 1e-8)
            )
        
        # Create scheduler
        scheduler_name = self.scheduler_config.get('name', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config.get('T_max', self.training_config['max_epochs']),
                eta_min=self.scheduler_config.get('eta_min', 0)
            )
        elif scheduler_name == 'step':
            scheduler = StepLR(
                optimizer,
                step_size=self.scheduler_config.get('step_size', 10),
                gamma=self.scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_name == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_config.get('factor', 0.5),
                patience=self.scheduler_config.get('patience', 5)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss'
                }
            }
        else:
            scheduler = None
        
        if scheduler:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        else:
            return optimizer
