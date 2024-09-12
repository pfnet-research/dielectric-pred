import os
import argparse
import yaml
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.training import DielectricModule, DielectricScalarModule
from data.dataset import JSONDataset
from models.equivariant_model import GatedEquivariantModel
from confidential.models import CombinedModel
from confidential.utils import load_pfp, collate_fn_dict


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    args = parser.parse_args()
    return args

def main(configs):
    seed = configs.pop("Seed")
    train_config = configs.pop("Train")
    model_config = configs.pop("Model")
    torch.manual_seed(seed)

    outdir = train_config.pop("save_path")
    os.makedirs(outdir, exist_ok=True)

    dataset_file = train_config.pop("dataset")
    target = train_config.pop("target")
    dataset = JSONDataset(
        json_file = dataset_file,
        target = target)

    keys = sorted(dataset.keys)
    print(f"{len(keys)} data are prepared!")

    # split dataset
    train_keys, test_keys = train_test_split(keys, test_size=0.1, random_state=seed)
    train_keys, val_keys = train_test_split(train_keys, test_size=0.1/0.9, random_state=seed)

    # prepare dataloader
    train_set = JSONDataset(
        dataset_file,
        target = target,
        keys = train_keys
    )
    val_set = JSONDataset(
        dataset_file,
        target = target,
        keys=val_keys
    )
    test_set = JSONDataset(
        dataset_file, 
        target = target,
        keys=test_keys
    )

    batch_size = train_config.pop("batch")
    num_workers = train_config.pop("num_workers")
    train_loader = DataLoader(
        train_set, batch_size=batch_size, collate_fn=collate_fn_dict, 
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, collate_fn=collate_fn_dict,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, collate_fn=collate_fn_dict,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # load the pre-trained PFP model
    pfp_layer = model_config.pop("pfp_layer")
    pfp_wrapped = load_pfp(
        load_parameters=True, 
        return_layer=pfp_layer)

    # Freeze PFP parameters
    train_pfp = model_config.pop("train_pfp") # Bool
    if train_pfp:
        pfp_wrapped.pfp.reset_parameters(4.0)
    else:
        for param in pfp_wrapped.parameters():
            param.requires_grad = False

    # Build Readout NN
    tensorial_model = GatedEquivariantModel(**model_config)
    # Build a combined model to connect readout NN to PFP
    model = CombinedModel(pfp_wrapped, tensorial_model)

    lr = train_config.pop("lr")
    pl_module = DielectricModule(
        model, 
        learning_rate=lr)
    
    project_name = f'{target}_layer{pfp_layer}_seed{seed}'
    wandb_logger = WandbLogger(project=project_name, job_type='train')

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=f"{outdir}/pl_checkpoints_seed{seed}/",
        filename="eps-{epoch:02d}-{val_loss:.2f}",
    )

    patience = train_config.pop("patience")
    earlystopping_callback = EarlyStopping("val_loss", mode="min", 
        patience=patience)

    epoch = train_config.pop("epoch")
    accelerator = train_config.pop("accelerator")
    devices = train_config.pop("device")
    clip_val = train_config.pop("gradient_clip")
    trainer = pl.Trainer(
        max_epochs=epoch, 
        accelerator=accelerator, 
        logger=wandb_logger,  
        devices=devices, 
        callbacks=[checkpoint_callback, earlystopping_callback],
        gradient_clip_val=clip_val,
        enable_model_summary=True)
    trainer.fit(
        model=pl_module, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader)

    trainer.test(
        model=pl_module, 
        dataloaders=test_loader,
        ckpt_path='best')


if __name__ == "__main__":
    args = parse()
    with open(args.config_file,'r') as f:
        configs = yaml.safe_load(f)
    main(configs)