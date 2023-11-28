import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import config
import torchvision
import pandas as pd
from PIL import Image
from model import FasterRCNNLightning
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2



columns = ["imgname", "classname", "class", "xmin", "ymin", "xmax", "ymax"]
data_df = pd.read_csv(config.ANNOTATION_FILE_PATH, sep=' ', header=None, names=columns,index_col=False)
data_df['full_path'] = config.MAIN_LOGO_FOLDER + '/' + data_df['imgname']



model = FasterRCNNLightning(num_classes=config.NUM_CLASSES,lr=config.LR)


cppath = '/checkpoints/BestFasterRCNN.ckpt'
checkpoint = torch.load(cppath)
model.load_state_dict(checkpoint['state_dict'])

def get_train_transform():
    return A.Compose([
        ToTensorV2(p=1.0)])
    
def apply_nms(orig_prediction, iou_thresh=0.3):
    
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction

def predplot_img_bbox(idx,iou_thresh=0.1):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    path = data_df.iloc[idx]['full_path']
    image = Image.open(path).convert("RGB")
    
    image = image.resize((config.WIDTH, config.HEIGHT))
    
    
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    
    a.imshow(image)
    plt.axis('off')
    
    img = np.array(image).astype(np.float32)
    img /= 255.0
    
    img = get_train_transform()(image=img)
    
    image_tensor = img['image'].unsqueeze(0) 
    
    model.eval()
    with torch.no_grad():
        prediction = model.forward(image_tensor.to(model.device))[0]
        
    prediction = apply_nms(prediction, iou_thresh=0.1)

            
    for box in (prediction['boxes']):
        x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth = 2,
                                 edgecolor = 'r',
                                 facecolor = 'none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)
    plt.show()