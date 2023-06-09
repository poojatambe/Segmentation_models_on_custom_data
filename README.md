# Segmentation_models_on_custom_data

## **Instance segmentation with yoloV8 and Segment Anything Model(SAM)**
 * Install:
 ```
 !pip install ultralytics
 !pip install git+https://github.com/facebookresearch/segment-anything.git
 !wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
 ```
 * Run the command:
 ```
 !python SAM_yolov8.py --img image_path --weights yolov8 weights --model_type SAM model type --checkpoints SAM model checkpoints
 ```
 Example:
 ```
 !python SAM_yolov8.py --img '/content/pexels-erick-15590648.jpg' --weights '/content/best.pt' --model_type 'vit_l' --checkpoints '/content/sam_vit_l_0b3195.pth'
 ```
 * Output:
 
 **yoloV8**
 
 ![yolov8_output](https://user-images.githubusercontent.com/64680838/236404508-36c272c9-7765-4925-b989-ce7858157099.jpg)
 
 **SAM_yoloV8**
![SAM_yolov8](https://user-images.githubusercontent.com/64680838/236404611-480a8d9f-898b-4bfd-b25e-a8980e713cb2.jpg)


* Reference:

1. https://github.com/ultralytics/ultralytics
2. https://github.com/facebookresearch/segment-anything
