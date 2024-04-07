import argparse
from pathlib import Path
import warnings

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets.augmentations import transform_normalize
from models.descriminator import PatchGANDiscriminator
from models.generator import Generator
from datasets.fran_dataset import FRANDataset
from training.trainer import FRAN

data_dir = Path('./data-demo/')

def get_args():
    parser = argparse.ArgumentParser(description='Train FRAN model.')
    parser.add_argument('--data_dir', '-C', type=str, default=data_dir, help='directory for data')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    image_meta = pd.read_csv(args.data_dir / "image_meta.csv")

    train_dataset = FRANDataset(image_meta, transform_normalize, args.data_dir / "synthetic_images")
    dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

    fran_model = FRAN(Generator(), PatchGANDiscriminator())
    fran_trainer = pl.Trainer(
        precision='16-mixed',
        devices=1,
        max_epochs=6,
        callbacks =[pl.callbacks.ModelCheckpoint(
            every_n_train_steps=5000,
            dirpath=args.data_dir,
            filename='fran-{step:05d}',
        )]
    ) 

    fran_trainer.fit(fran_model, dataloader)