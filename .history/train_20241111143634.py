import pytorch_lightning as pl
from nif_torch.models.nif import NIFModule
from nif_torch.data.dataset import NIFDataModule

def train_nif(cfg):
    # Initialize model and data
    model = NIFModule(cfg['shape_net'], cfg['parameter_net'])
    datamodule = NIFDataModule(
        data_path=cfg['data_path'],
        n_feature=cfg['n_feature'],
        n_target=cfg['n_target'],
        batch_size=cfg['batch_size']
    )
    
    # Initialize trainer without DDP strategy
    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        accelerator='mps',
        devices=1,
        precision='32',
        logger=True,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath='checkpoints',
                filename='{epoch}-{train_loss:.2f}',
                save_top_k=3,
                monitor='train_loss'
            )
        ]
    )
    
    # Train the model
    trainer.fit(model, datamodule)
