import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pytorch_lightning as pl


# Lightning Module
class FasterRCNNLightning(pl.LightningModule):
    def __init__(self, num_classes, lr):
        super(FasterRCNNLightning, self).__init__()
        
        self.lr = lr
        self.validation_step_outputs = []

        # Load a pre-trained Faster R-CNN model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)

        # Modify the classifier head for your number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Create the Flickr27Dataset instance

    def forward(self, x, target=None):
        if target is not None:
            return self.model(x, target)
        else:
            return self.model(x)


    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images, targets)
        loss = sum(loss for loss in outputs.values())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    