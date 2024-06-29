from typing import Dict, List
from ase.io import read
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import pytorch_lightning as pl
import torch
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
import torch.nn.functional as F


from confidential import get_pfp_inputs_dict

class DielectricModule(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate: float = 1e-4,
        optimizer: Optimizer = None,
        scheduler = 'plateau',
        decay_steps: int = 1000,
        decay_alpha: float = 0.01,
        reduce_lr_patience: int = 10,
        reduce_lr_factor: float = 0.8,
        ):
        """Init DielectricModel with key parameters
        
        Args:
            model: the combined model of PFP and Equivariant Readout NN
            learning_rate: learning rate for training
            optimizer: optimizer for training
            scheduler: scheduler for training
            decay_steps: number of steps for decaying learning rate for CosineAnnealingLR
            decay_alpha: parameter determines the minimum learning rate for CosineAnnealingLR
            reduce_lr_patience: number of allowed epochs with no improvement after which the learning rate will be reduced
            reduce_lr_factor: factor by which the learning rate will be reduced
        """
        super().__init__()

        self.model = model
        self.lr = learning_rate
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.decay_steps = decay_steps
        self.decay_alpha = decay_alpha
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor

        self.l1loss = nn.L1Loss(reduction="none")
        self.mseloss = nn.MSELoss(reduction="none")


    def forward(
        self, 
        batch):

        return self.model(**batch)

    def loss_fn(
        self,
        pred_y: Tensor,
        y: Tensor) -> Dict: 
        """Define the loss function
        Args:
            pred_y: Predicted dielectric tensor, shape: (batch x 3 x 3)
            y: True dielectric tensor, shape: (batch x 3 x 3)
        Returns:
            loss_dict: losses
        """

        loss_trace = self.mseloss(
            pred_y[:, [0,1,2], [0,1,2]].mean(axis=1), 
            y[:, [0,1,2], [0,1,2]].mean(axis=1),
        ).mean(0)   # (batch,)
        loss_diag = self.mseloss(
            pred_y[:, [0,1,2], [0,1,2]], 
            y[:, [0,1,2], [0,1,2]],
        ).sum(1).mean(0)   # (batch, 3)
        loss_off = self.mseloss(
            pred_y[:, [0, 0, 1], [1, 2, 2]],
            y[:, [0, 0, 1], [1, 2, 2]]
        ).sum(1).mean(0)   # (batch, 3)

        loss_tensor = self.mseloss(pred_y, y).view(-1, 9).sum(1).mean(0)    # (batch, 3, 3)

        loss = loss_tensor
        
        loss_dict ={
            "loss": loss,
            "trace_loss": loss_trace,
            "diag_loss": loss_diag,
            "off_loss": loss_off,
            "tensor_loss": loss_tensor
        }

        return loss_dict


    def training_step(
        self,
        batch: Tensor,
        batch_idx: int) -> Tensor:
        """Training step
        Args:
            batch: Data batch
            batch_idx: Batch index
        Returns:
            Training loss
        """
        labels = batch.pop("label")
        ids = batch.pop("id")
        batch_size = labels.size(0)

        results = self(batch)

        loss_dict = self.loss_fn(results, labels)

        train_loss = {
            "train_"+key: val 
            for key, val in loss_dict.items()
        }
        train_loss["lr"] = self.trainer.optimizers[0].param_groups[0]["lr"]

        self.log_dict(train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        return train_loss["train_loss"]

    def validation_step(
        self,
        batch: Tensor,
        batch_idx: int) -> Tensor:
        """Validation step
        Args:
            batch: Data batch
            batch_idx: Batch index
        Returns:
            Validation loss
        """        
        labels = batch.pop("label")
        ids = batch.pop("id")
        batch_size = labels.size(0)

        results = self(batch)
        loss_dict = self.loss_fn(results, labels)

        val_loss = {
            "val_"+key: val 
            for key, val in loss_dict.items()
        }

        self.log_dict(val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        
        return val_loss["val_loss"]

    def test_step(
        self,
        batch: Tensor,
        batch_idx: int):
        """Test step
        Args:
            batch: Data batch
            batch_idx: Batch index
        """        
        labels = batch.pop("label")     # (batch, 3, 3)
        ids = batch.pop("id")
        batch_size = labels.size(0)

        results = self(batch)         # (batch, 3, 3)

        loss = F.l1_loss(
            labels[:, [0,0,0,1,1,2], [0,1,2,1,2,2]],
            results[:, [0,0,0,1,1,2], [0,1,2,1,2,2]],
            reduction='none').mean()

        self.log_dict({'loss': loss}, batch_size=batch_size)

        return {'loss': loss}

    def configure_optimizers(self):
        """Configure the optimizer
        """
        if self.optimizer is None:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                eps=1e-8,
            )
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                eps=1e-8,
            )
        else:
            optimizer = self.optimizer
        if self.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.decay_steps,
                eta_min=self.lr * self.decay_alpha,
            )
        elif self.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                patience=self.reduce_lr_patience, 
                factor=self.reduce_lr_factor
        )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": "val_loss"},
                }
        else:
            scheduler = self.scheduler
        return [optimizer,], [scheduler,]

    @torch.no_grad()
    def predict_atoms_from_file(
        self,
        structure_file: str):

        self.model.eval()

        if isinstance(structure_file, str):
            atoms_batch = [read(structure_file)]
        elif isinstance(structure_file, list):
            atoms_batch = [read(f) for f in structure_file]
        else:
            raise TypeError(f"Input structure_file should be str or list, not {type(structure_file)}!")
        
        results = []
        for atoms in atoms_batch:
            # convert to conventional structures
            structure = AseAtomsAdaptor.get_structure(atoms)
            group_analyzer = SpacegroupAnalyzer(structure)
            conv = group_analyzer.get_refined_structure()
            atoms = AseAtomsAdaptor.get_atoms(conv)

            pred = self.predict_atoms(atoms)
            results.append(pred)

        results = torch.cat(results, dim=0)
        return results.detach().cpu()

    @torch.no_grad()
    def predict_atoms_from_structures(
        self,
        structures: List[Atoms]):
        self.model.eval()
        
        results = []
        for atoms in structures:
            pred = self.predict_atoms(atoms)
            results.append(pred)

        results = torch.cat(results, dim=0)
        return results.detach().cpu()    

    @torch.no_grad()
    def predict_atoms(
        self,
        atoms: Atoms):

        inputs = get_pfp_inputs_dict(atoms)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return self.model(**inputs)