# Object Detection
This is just an implementation of objection detection in images and videos using torchvision from Pytorch.

## Installation
Install the latest version of Pytorch, OpenCV, Pillow and python

### Running the tests
1. Clone the repository

2. Follow the given steps
```
import torchvision
import torch
import obj_det

detection = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
detection.eval()
``` 
3. For object detection in images
```
pil = obj_det.image_predictions(detection, 'lockdown.jpg', .8)
``` 
4. For detection in videos
```
obj_det.video_detection(detection, 'new.mp4', 'ou.avi', 50, .9)
``` 
