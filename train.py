from model import FasterRCNNLightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloader import Flickr27DataModule
from lightning.pytorch import seed_everything
import config


seed_everything(42, workers=True)


model = FasterRCNNLightning(num_classes=config.NUM_CLASSES,lr=config.LR)


checkpoint_callback = ModelCheckpoint(
    dirpath = 'checkpoints',
    filename = config.CHECKPOINT_NAME,
    save_top_k = 1,
    verbose = True,
    monitor = 'train_loss',
    mode = 'min'
)

data_module = Flickr27DataModule(root_folder=config.MAIN_LOGO_FOLDER,
                                annotation_file=config.ANNOTATION_FILE_PATH,
                                batch_size=config.BATCH_SIZE,
                                val_split=config.VAL_SPLIT)

data_module.setup()

trainer = pl.Trainer(devices=-1, 
                  accelerator="gpu",
                  check_val_every_n_epoch=5,
                  callbacks=[checkpoint_callback],
                  max_epochs=config.MAX_EPOCHS)


trainer.fit(model=model,datamodule=data_module)
