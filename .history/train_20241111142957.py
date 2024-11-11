import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
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
    
    # Initialize trainer with distributed strategy
   python3 -c "import torch; print(torch.backends.mps.is_available())"
    
    # Train the model
    trainer.fit(model, datamodule)
