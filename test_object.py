from jetbot import ObjectDetector
from jetbot import Camera
import torch
import torchvision
import torch.nn.functional as F
import cv2
import numpy as np
from jetbot import Robot
from jetbot import bgr8_to_jpeg
import time


def preprocess(camera_value):
    global device, normalize
    x = camera_value
    x = cv2.resize(x, (224, 224))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x

def detection_center(detection):
    """Computes the center x, y coordinates of the object"""
    bbox = detection['bbox']
    center_x = (bbox[0] + bbox[2]) / 2.0 - 0.5
    center_y = (bbox[1] + bbox[3]) / 2.0 - 0.5
    return (center_x, center_y)

def norm(vec):
    """Computes the length of the 2D vector"""
    return np.sqrt(vec[0] ** 2 + vec[1] ** 2)

def closest_detection(detections):
    """Finds the detection closest to the image center"""
    closest_detection = None
    for det in detections:
        center = detection_center(det)
        if closest_detection is None:
            closest_detection = det
        elif norm(detection_center(det)) < norm(detection_center(closest_detection)):
            closest_detection = det
    return closest_detection

def execute(change):
    image = change['new']

    # execute collision model to determine if blocked
    collision_output = collision_model(preprocess(image)).detach().cpu()
    prob_blocked = float(F.softmax(collision_output.flatten(), dim=0)[0])
    blocked_widget.value = prob_blocked

    # turn left if blocked
    if prob_blocked > 0.5:
        robot.left(0.3)
        image_widget.value = bgr8_to_jpeg(image)
        return

    # compute all detected objects
    detections = model(image)

    # draw all detections on image
    for det in detections[0]:
        bbox = det['bbox']
        cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])),
                      (int(width * bbox[2]), int(height * bbox[3])), (255, 0, 0), 2)

    # select detections that match selected class label
    matching_detections = [d for d in detections[0] if d['label'] == int(label_widget.value)]

    # get detection closest to center of field of view and draw it
    det = closest_detection(matching_detections)
    if det is not None:
        bbox = det['bbox']
        cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])),
                      (int(width * bbox[2]), int(height * bbox[3])), (0, 255, 0), 5)

    # otherwise go forward if no target detected
    if det is None:
        robot.forward(float(speed_widget.value))

    # otherwsie steer towards target
    else:
        # move robot forward and steer proportional target's x-distance from center
        center = detection_center(det)
        robot.set_motors(
            float(speed_widget.value + turn_gain_widget.value * center[0]),
            float(speed_widget.value - turn_gain_widget.value * center[0])
        )

    # update image widget
    image_widget.value = bgr8_to_jpeg(image)


model = ObjectDetector('/home/dewald/projects/frameworks/nvidia/deepstream_sdk_v4.0.2_jetson/samples/models/SSD-Mobilenet-v2/ssd_mobilenet_v2_coco.engine')
camera = Camera.instance(width=300, height=300)
detections = model(camera.value)
print(detections)

collision_model = torchvision.models.alexnet(pretrained=False)
collision_model.classifier[6] = torch.nn.Linear(collision_model.classifier[6].in_features, 2)
collision_model.load_state_dict(torch.load('/home/dewald/projects/jetbot/jetbot/best_model.pth'))
device = torch.device('cuda')
collision_model = collision_model.to(device)
mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])
normalize = torchvision.transforms.Normalize(mean, stdev)

# Robot Movement
robot = Robot()

# Widgets
blocked_widget = widgets.FloatSlider(min=0.0, max=1.0, value=0.0, description='blocked')
image_widget = widgets.Image(format='jpeg', width=300, height=300)
label_widget = widgets.IntText(value=1, description='tracked label')
speed_widget = widgets.FloatSlider(value=0.4, min=0.0, max=1.0, description='speed')
turn_gain_widget = widgets.FloatSlider(value=0.8, min=0.0, max=2.0, description='turn gain')

display(widgets.VBox([
    widgets.HBox([image_widget, blocked_widget]),
    label_widget,
    speed_widget,
    turn_gain_widget
]))

width = int(image_widget.width)
height = int(image_widget.height)

execute({'new': camera.value})

# Loop
camera.unobserve_all()
camera.observe(execute, names='value')

# Stop
camera.unobserve_all()
time.sleep(1.0)
robot.stop()


    

