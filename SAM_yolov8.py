# Yolov8 bounding box output as prompt to SAM model for segmentation.

from ultralytics import YOLO
from IPython.display import display, Image
import numpy as np
import cv2
# from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
import argparse

class segment:
  def __init__(self, img, model, model_type, checkpoint):
    self.img= img
    self.model= model
    self.model_sam= model_type
    self.checkpoint= checkpoint

  def __call__(self):
    results= self.yolov8()
    self.SAM(results[0].boxes.xyxy)
  
  def yolov8(self):
    model= YOLO(self.model)
    results = model.predict(source=self.img) 
    res_plotted = results[0].plot()
    cv2.imwrite('yolov8_output.jpg',res_plotted)
    return results

  def SAM(self, output_box):
    img= cv2.imread(self.img)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sam = sam_model_registry[self.model_sam](checkpoint=self.checkpoint)
    # Initialize SAM model
    predictor = SamPredictor(sam)
    # Generate input image embeddings.
    predictor.set_image(img)
    input_boxes = output_box
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, img.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    for mask in masks:
        self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box in input_boxes:
        self.show_box(box.cpu().numpy(), plt.gca())
    # plt.axis('off')
    # plt.show()
    plt.savefig('SAM_yolov8.jpg')
    
  def show_mask(self, mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)  
    
  def show_box(self, box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

if __name__ == "__main__":
  parser= argparse.ArgumentParser()
  parser.add_argument('--img', help='Specify input image path', default='', type=str)
  parser.add_argument('--weights', default='./best.pt', type=str, help='Specify object detection weights')
  parser.add_argument('--model_type', default='vit_l', type= str)
  parser.add_argument('--checkpoints', default='', type= str)
  opt= parser.parse_args()
  sam= segment(opt.img, opt.weights, opt.model_type, opt.checkpoints)
  sam()
  