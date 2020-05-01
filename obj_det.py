from torchvision import transforms
from PIL import Image
import cv2

<<<<<<< HEAD
=======
#img_path = 'lockdown.jpg'
#detection = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#detection.eval()

>>>>>>> 6d946e7796e4bb04686f783d31dcaa9e28b127fc
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

<<<<<<< HEAD

def get_prediction(model, img_path, threshold):
    
    if type(img_path) != str:    
        transform = transforms.Compose([transforms.ToTensor()]) # Defing PyTorch Transform
        img = transform(img_path) # Apply the transform to the image
        pred = model([img]) # Pass the image to the model
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        pred_score = pred_score[:pred_t+1]
  
    else:
        img = Image.open(img_path)
        transform = transforms.Compose([transforms.ToTensor()]) # Defing PyTorch Transform
        img = transform(img) # Apply the transform to the image
        pred = model([img]) # Pass the image to the model
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        pred_score = pred_score[:pred_t+1]
      
    return pred_boxes, pred_class, pred_score
=======
img = Image.open(img_path) # Load the image

def get_prediction(model, img, threshold):
  """
  Returns the predictions in terms of boxes, class and scores
  """
  transform = transforms.Compose([transforms.ToTensor()]) # Defing PyTorch Transform
  img = transform(img) # Apply the transform to the image
  pred = model([img]) # Pass the image to the model
  pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
  pred_score = list(pred[0]['scores'].detach().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  pred_score = pred_score[:pred_t+1]
  return pred_boxes, pred_class, pred_score
>>>>>>> 6d946e7796e4bb04686f783d31dcaa9e28b127fc

def return_one(bx, cl, sc):
  for i,j in enumerate(bx):
    bx[i].append(cl[i])
    bx[i].append(sc[i])
  return bx

def object_detection(bx, img_path):
  """
  Returns the image with bounding boxes, scores and classes
  """
  img = cv2.imread(img_path)

  for i, j in enumerate(bx):
    cv2.rectangle(img, bx[i][0], bx[i][1], (25, 70, 200), 1)
    cv2.putText(img, bx[i][2], bx[i][1], cv2.FONT_ITALIC, 1, (255,0,255),thickness = 1)
    cv2.putText(img, str(bx[i][3]), bx[i][0], cv2.FONT_ITALIC, 1, (255,0,255),thickness = 1)
  pil = Image.fromarray(img)
  return pil

def image_predictions(model, img_path,threshold):
  """
  Return the combined output of images
  """
  img = Image.open(img_path)
  pre_bx, pre_cl, pre_score = get_prediction(model, img, threshold)
  all_ = return_one(pre_bx, pre_cl, pre_score)
  pil_img = object_detection(all_, img_path)

  return pil_img

def video_detection(model, input_file, output_file, fps=30, score_filter=0.6):
    """
    Returns detections from video and save them in separate files
    """
    # Read in the video
    video = cv2.VideoCapture(input_file)

    # Video frame dimensions
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scale down frames when passing into model for faster speeds
    scaled_size = 800
    scale_down_factor = min(frame_height, frame_width) / scaled_size

    # The VideoWriter with which we'll write our video with the boxes and labels
    # Parameters: filename, fourcc, fps, frame_size
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width, frame_height))

    # Transform to apply on individual frames of the video
    transform_frame = transforms.Compose([  # TODO Issue #16
        transforms.ToPILImage(),
        transforms.Resize(scaled_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Loop through every frame of the video
    while True:
        ret, frame = video.read()
        # Stop the loop when we're done with the video
        if not ret:
            break

        # The transformed frame is what we'll feed into our model
        transformed_frame = transform_frame(frame)
        #transformed_frame = frame  # TODO: Issue #16
        box, label, score = get_prediction(model, frame, score_filter)
        all_pred = return_one(box, label, score)
        # Add the top prediction of each class to the frame
        for i, j in enumerate(all_pred):
            if all_pred[i][3] < score_filter:
                continue

            # Since the predictions are for scaled down frames,
            # we need to increase the box dimensions
            # box *= scale_down_factor  # TODO Issue #16

            # Create the box around each object detected
            # Parameters: frame, (start_x, start_y), (end_x, end_y), (r, g, b), thickness
            cv2.rectangle(frame, all_pred[i][0], all_pred[i][1], (255, 0, 0), 3)

            # Write the label and score for the boxes
            # Parameters: frame, text, (start_x, start_y), font, font scale, (r, g, b), thickness
            cv2.putText(frame, '{}: {}'.format(all_pred[i][2], round(all_pred[i][3].item(), 2)), (all_pred[i][0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # Write this frame to our video file
        out.write(frame)

        # If the 'q' key is pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # When finished, release the video capture and writer objects
    video.release()
    out.release()

    # Close all the frames
    cv2.destroyAllWindows()
