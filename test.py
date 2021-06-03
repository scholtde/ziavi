#!/usr/bin/env python

"""
Copyright (c) 2020, Rhizoo Christos Technologies. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import serial
import time
import random
import os
import time
import pygame, sys
import pygame.locals
import RPi.GPIO as GPIO
from uuid import uuid1
from jetbot import ObjectDetector
#from jetbot import Camera
from packages import Camera
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from jetbot import bgr8_to_jpeg

from flask import Response
from flask import Flask
from flask import render_template
import threading
import random


# Functions
def train_bot():
    # Create dataset instance
    dataset = datasets.ImageFolder(
        'dataset',
        transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    # Split dataset into train and test sets
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 50, 50])

    # Create data loaders to load data in batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )

    # Define the neural network
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )

    t_model = models.alexnet(pretrained=True)  # Download if not exist
    t_model.classifier[6] = torch.nn.Linear(t_model.classifier[6].in_features, 2)

    device = torch.device('cuda')
    t_model = t_model.to(device)

    # Train the neural network
    NUM_EPOCHS = 50
    BEST_MODEL_PATH = 'models/classification/best_model.pth'
    best_accuracy = 0.0

    optimizer = optim.SGD(t_model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(NUM_EPOCHS):

        for images, labels in iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = t_model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

        test_error_count = 0.0
        for images, labels in iter(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))

        test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
        print('%d: %f' % (epoch, test_accuracy))
        if test_accuracy > best_accuracy:
            torch.save(t_model.state_dict(), BEST_MODEL_PATH)
            best_accuracy = test_accuracy


def prepareUI(flag):
    global joystick
    global windowSurface
    global Image
    global BACKG	
    pygame.display.set_caption('ZIAVI v.1.0')
    if flag == 0:
        WIDTH = 1024
        HEIGHT = 750
        windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN, 32)
    elif flag == 1:
        WIDTH = 510
        HEIGHT = 750
        windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
 
    Icon = pygame.image.load('/home/dewald/projects/ziavi/img/RCT_Logo_205x71l.png')
    BACKG = (0,0,0)
    x = 90 # (WIDTH * 0.90)
    y = 12 # (HEIGHT * 0.80)
    windowSurface.fill(BACKG)
    windowSurface.blit(Icon, (x,y))
    pygame.display.set_icon(Icon)
    pygame.display.update()
    pygame.display.flip()

    # Inits
    #pygame.mixer.init()
    #pygame.mixer.music.load('/home/dewald/projects/ziavi/audio/alert_1.wav')
    #pygame.mixer.music.play()
    #while pygame.mixer.music.get_busy() == True:
    #  continue
    pygame.joystick.init()
    joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
    # Get count of joysticks
    joystick_count = pygame.joystick.get_count()
 
    print("Number of joysticks: {}".format(joystick_count))
 
    # For each joystick:
    for i in range(joystick_count):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()
 
        print("Joystick {}".format(i))
         
        # Get the name from the OS for the controller/joystick
        name = joystick.get_name()
        print("Joystick name: {}".format(name))
 
        # Usually axis run in pairs, up/down for one, and left/right for
        # the other.
        axes = joystick.get_numaxes()
        print("Number of axes: {}".format(axes))
         
        for i in range(axes):
            axis = joystick.get_axis(i)
            print("Axis {} value: {:>6.3f}".format(i, axis))
         
        buttons = joystick.get_numbuttons()
        print("Number of buttons: {}".format(buttons))
         
        for i in range(buttons):
            button = joystick.get_button(i)
            print("Button {:>2} value: {}".format(i, button))
         
        # Hat switch. All or nothing for direction, not like joysticks.
        # Value comes back in an array.
        hats = joystick.get_numhats()
        print("Number of hats: {}".format(hats))
 
        for i in range(hats):
            hat = joystick.get_hat(i)
            print("Hat {} value: {}".format(i, str(hat)))

         
def writeText(path, text, xoffset, x, y, size, w, h, textCol, rectCol):
    switcher = {
      0: "/home/dewald/projects/ziavi/fonts/LCARS.ttf",
      1: "/home/dewald/projects/ziavi/fonts/LCARSGTJ2.ttf",
      2: "/home/dewald/projects/ziavi/fonts/LCARSGTJ3.ttf",
    }
    #windowSurface.fill(BACKG) 
    #windowSurface.blit(Image, (x,y))
    Rectangle = pygame.Rect(x-5, y, w, h*1.1)
    pygame.draw.rect(windowSurface,(rectCol),Rectangle,0) 
    font = pygame.font.Font(switcher.get(path, "/home/dewald/projects/ziavi/fonts/LCARSGTJ2.ttf"), size)
    label = font.render(text, 1, (textCol))
    windowSurface.blit(label,(x+xoffset,y))
    #pygame.display.flip()
    pygame.display.update(Rectangle)	
    return Rectangle


def Labels():
    #(path, text, xoffset, x, y, size, w, h, textCol, recCol)
    y = 12
    x = 300
    offset = 0
    rec = 0,0,0
    writeText(1, "ZIAVI", offset, x, y, 50, 150, 50, (255, 255, 255), rec)
    y = 55
    writeText(1, "Version v1.0", offset, x, y, 20, 150, 20, (29,136,133), rec) 

    y = 110
    x = 19
    size = 30
    w = 170
    offset = 43
    rec = 0, 77, 77
    txtCol = 255,255,255

    btnAL = writeText(1, "X-value", 95, x, y, size, w, size, txtCol, rec)
    y = y+offset
    btnVT = writeText(1, "Y-value", 95, x, y, size, w, size, txtCol, rec)
    y = y+offset
    btnHT = writeText(1, "G-CODE", 105, x, y, size, w, size, txtCol, rec)
    y = y+offset
    btnCT = writeText(1, "STATUS", 100, x, y, size, w, size, txtCol, rec)
 
    x = 12
    y = 275
    size = 55
    w = 200
    offset = 60
    rec = 0,0,0
    txtCol = 255,255,255
    writeText(1, "MODE:", 80, x, y, size, w, size, txtCol, rec)
    y = y+offset
    writeText(1, "PATH:", 85, x, y, size, w, size, txtCol, rec)
    y = y+offset
    writeText(1, "DIRECTION:", 15, x, y, size, w, size, txtCol, rec)

    x = 12
    y = 460 
    #windowSurface.blit(image, (x, y))
    Rectangle = pygame.Rect(x, y, 480, 270)
    pygame.draw.rect(windowSurface,(50, 50, 50),Rectangle,0) 
    pygame.display.update(Rectangle)


def save_snapshot(directory):
    image_path = os.path.join(directory, str(uuid1()) + '.jpg')
    # save image
    image = camera.image_array
    image = cv2.resize(image, (224, 224))
    status = cv2.imwrite(image_path, image)
    print("Image written to file-system : ", status)
    #with open(image_path, 'wb') as f:
    #    f.write(image.image_array)


def save_free():
    global free_dir, free_count
    save_snapshot(free_dir)
    free_count = len(os.listdir(free_dir))
    print("Free    Count : ", free_count)
    print("Blocked Count : ", blocked_count)

    
def save_blocked():
    global blocked_dir, blocked_count
    save_snapshot(blocked_dir)
    blocked_count = len(os.listdir(blocked_dir))
    print("Free    Count : ", free_count)
    print("Blocked Count : ", blocked_count)


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


def pause(c_time, delay):
    if time.time() - c_time > delay:
        return True
    else:
        return False

def move_robot(x, y, sp):
    global g_busy

    t_y = 110
    t_x = 200
    size = 30
    w = 250
    offset = 43
    rec = 0, 77, 77
    txtCol = 255,255,255

    btnAL = writeText(1, str(round(x,2)), 5, t_x, t_y, size, w, size, txtCol, rec)
    t_y = t_y + offset
    btnVT = writeText(1, str(round(y,2)), 5, t_x, t_y, size, w, size, txtCol, rec)
    
    gcode = "$J=X"+str(round(x,2))+" Y"+str(round(y,2))+" F"+str(sp)
    l_block = bytes(gcode, 'utf-8')
    #print("SND: " + str(l_count) + " : " + str(l_block))
    t_y = t_y + offset
    btnHT = writeText(1, gcode, 5, t_x, t_y + 5, 20, w, size, txtCol, rec)
    s.write(l_block + b'\n') # Send g-code block to grbl
    g_busy = True
    out_temp = s.readline().strip() # Wait for grbl response
    if out_temp.find(b'ok') < 0 and out_temp.find(b'error') < 0 :
        pass
    else:
        g_busy = False
        #print("  Debug: ",out_temp) # Debug response
        t_y = t_y + offset
        btnCT = writeText(1, out_temp, 5, t_x, t_y, size, w, size, txtCol, rec)

 
def execute(change):
    global j_x
    global j_y
    global g_stop
    global check_time_blocked
    global check_time_free
    global outputFrame, lock

    image = change['new']

    # execute collision model to determine if blocked
    collision_output = collision_model(preprocess(image)).detach().cpu()
    prob_blocked = float(F.softmax(collision_output.flatten(), dim=0)[0])
    #blocked_widget.value = prob_blocked

    follow = True  # Follow object if robot is not blocked
    mode = "MANUAL"
    path = "n/a"
    direction = "STOPPED"

    if g_stop is False:
        mode = "AUTO"
        # turn right if blocked
        if prob_blocked >= 0.50:
            follow = False
            path = str(round(prob_blocked * 100, 1)) + "% - BLOCKED"
            direction = "RIGHT"
            if pause(check_time_blocked, 0.7):
                #print("robot is bloked! - moving BACKWARDS")
                j_x -= 0.15
                j_y -= 0.15
                speed = 200
                move_robot(j_x, j_y, speed)

                #print("robot is bloked! - moving RIGHT")
                j_x += 0.5
                j_y -= 0.5
                speed = 200
                move_robot(j_x, j_y, speed)

                check_time_blocked = time.time()
            
        # If robot is not blocked, move towards target
        else:# and pause(check_time_free, 0.5):
            path = str(round(prob_blocked * 100, 1)) + "% - FREE"
            direction = "FORWARD"
            # compute all detected objects
            detections = model(image)

            # draw all detections on image
            for det in detections[0]:
                bbox = det['bbox']
                cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])),
                          (int(width * bbox[2]), int(height * bbox[3])), (255, 0, 0), 2)

            # select detections that match selected class label
            matching_detections = [d for d in detections[0] if d['label'] == int("1")]

            # get detection closest to center of field of view and draw it
            det = closest_detection(matching_detections)
            if det is not None:
                direction = "DANCE"
                bbox = det['bbox']
                cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])),
                          (int(width * bbox[2]), int(height * bbox[3])), (0, 255, 0), 5)


            check_time_free = time.time()

            # otherwise go forward if no target detected
            if det is None:
                #print("robot is moving forward..")
                j_x += 0.23
                j_y += 0.23
                speed = 200
                move_robot(j_x, j_y, speed)

            # otherwsie steer towards target
            else:
                # move robot forward and steer proportional target's x-distance from center
                center = detection_center(det)
                # robot.set_motors(
                #     float(speed_widget.value + turn_gain_widget.value * center[0]),
                #     float(speed_widget.value - turn_gain_widget.value * center[0])
                # )

                #j_x += random.randint(-1,1)*0.25
                #j_y -= random.randint(-1,1)*0.25
                
                print(center[0])
                if -0.2 < center[0] < 0.2:
                    j_x += 0.2 + 0.5*center[0]
                    j_y += 0.2 - 0.5*center[0]
                    speed = 200  #str(random.randint(200,250))
                    move_robot(j_x, j_y, speed)
                elif center[0] >= 0.2:
                    j_x += 0.2
                    j_y -= 0.2
                    speed = 200  #str(random.randint(200,250))
                    move_robot(j_x, j_y, speed)
                elif center[0] <= -0.2:
                    j_x -= 0.2
                    j_y += 0.2
                    speed = 200  #str(random.randint(200,250))
                    move_robot(j_x, j_y, speed)
    
    x = 200
    y = 275
    size = 55
    w = 270
    offset = 60
    rec = 0,0,0
    txtCol = 255,255,255
    writeText(1, mode, 5, x, y, size, w, size, txtCol, rec)
    y = y+offset
    writeText(1, path, 5, x, y, size, w, size, txtCol, rec)
    y = y+offset
    writeText(1, direction, 5, x, y, size, w, size, txtCol, rec)

    # update image
    #image_widget.value = bgr8_to_jpeg(image)
    image = cv2.resize(image, (480, 270))
    # acquire the lock, set the output frame, and release the
    # lock
    with lock:
        outputFrame = image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.swapaxes(0, 1)
    image = pygame.surfarray.make_surface(image)
    x = 12
    y = 460 
    windowSurface.blit(image, (x, y))
    Rectangle = pygame.Rect(x, y, 480, 270)
    pygame.display.update(Rectangle)


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


def main():
    # Globals
    global l_count
    global g_count
    global c_line
    global j_x
    global j_y
    global g_busy
    global g_stop
    
    while True:
      
        # Check Ctrl+C
        try:
            #detections = model(camera.value)
            #print(detections)
            execute({'new': camera.image_array})
           
        except KeyboardInterrupt:
            camera.stop()
            s.close()
            #p.stop()
            #GPIO.cleanup()
            pygame.quit()
            sys.exit()

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    #pi.stop()
                    #ser1.close()
                    #print(">> Serial Port " + ser1.port + " closed")
                    #pygame.mixer.music.load('/home/dewald/projects/ziavi/audio/deactivation_complete.wav')
                    #pygame.mixer.music.play()
                    #while pygame.mixer.music.get_busy() == True:
                    #  continue
                    camera.stop()
                    s.close()
                    #p.stop()
                    #GPIO.cleanup()
                    pygame.quit()
                    sys.exit()

                if event.key == pygame.K_r:
                    camera.exec_rotate()

                #if event.key == pygame.K_MINUS:
                #    camera.exec_rotate()

            # Possible joystick actions: JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN
            # JOYBUTTONUP JOYHATMOTION
            if event.type == pygame.JOYBUTTONDOWN:
                print("Joystick button pressed.")
                buttons = joystick.get_numbuttons()
                for i in range(buttons):
                    button = joystick.get_button(i)
                    print("Button {:>2} value: {}".format(i, button))

                    if i == 0 and button:
                        print("UP - is pressed")
                        g_stop = True

                        j_x += 0.25
                        j_y += 0.25
                        speed = 200
                        move_robot(j_x, j_y, speed)

                    if i == 1 and button:
                        print("RIGHT - is pressed")
                        g_stop = True

                        j_x += 0.25
                        j_y -= 0.25
                        speed = 200
                        move_robot(j_x, j_y, speed)

                    if i == 2 and button:
                        print("DOWN - is pressed")
                        g_stop = True

                        j_x -= 0.25
                        j_y -= 0.25
                        speed = 200
                        move_robot(j_x, j_y, speed)

                    if i == 3 and button:
                        print("LEFT - is pressed")
                        g_stop = True

                        j_x -= 0.25
                        j_y += 0.25
                        speed = 200
                        move_robot(j_x, j_y, speed)

                    if i == 4 and button:
                        g_stop = True
                        time.sleep(1)
                        print("L.UPPER - is pressed")
                        save_free()

                    if i == 5 and button:
                        print("R.UPPER - is pressed")
                        g_stop = True

                        # Stop all JOG commands first
                        time.sleep(0.5)
                        l_block = bytes('!', 'ascii') # Feed HOLD command
                        print("SND: " + str(l_block))
                        s.write(l_block) # Send g-code block to grbl
                        out_temp = s.readline().strip() # Wait for grbl response
                        if out_temp.find(b'ok') < 0 and out_temp.find(b'error') < 0 :
                            pass
                        else:
                            g_busy = False
                            print("  Debug: ", out_temp)  # Debug response
                        time.sleep(0.5)

                    if i == 6 and button:
                        g_stop = True
                        time.sleep(1)
                        print("L.LOWER - is pressed")
                        save_blocked()

                    if i == 7 and button:
                        print("R.LOWER - is pressed")
                        g_stop = False
                        time.sleep(0.5)

                    if i == 8 and button:
                        print("SELECT - is pressed")
                        g_stop = False
                        print("starting training mode..")
                        time.sleep(10)
                        print("training complete!")

                    if i == 9 and button:
                        print("START - is pressed")
                        l_count += 1 # Iterate line counter
                        j_x += random.randint(-1,1)
                        j_y -= random.randint(-1,1)
                        j_speed = str(random.randint(200,250))
                        l_block = bytes("$J=X"+str(j_x)+" Y"+str(j_y)+" F"+j_speed, 'utf-8')
                        c_line.append(len(l_block)+1) # Track number of characters in grbl serial read buffer
                        grbl_out = b''

                        while sum(c_line) >= RX_BUFFER_SIZE-1 | s.inWaiting() :
                            out_temp = s.readline().strip() # Wait for grbl response
                            if out_temp.find(b'ok') < 0 and out_temp.find(b'error') < 0 :
                                print("  Debug: ",out_temp) # Debug response
                            else :
                                grbl_out += out_temp;
                                g_count += 1 # Iterate g-code counter
                                grbl_out += bytes(str(g_count), 'utf-8') # Add line finished indicator
                                del c_line[0] # Delete the block character count corresponding to the last 'ok'
                        if verbose:
                            print("SND: " + str(l_count) + " : " + str(l_block))
                            s.write(l_block + b'\n') # Send g-code block to grbl
                        if verbose :
                            print("BUF:",str(sum(c_line)),"REC:",grbl_out)

                        time.sleep(0.1)

            if event.type == pygame.JOYBUTTONUP:
                print("Joystick button released.")


# Load UI
print("Loading UI")
pygame.init()
prepareUI(1) 
Labels()

# Web UI (flask)
outputFrame = None
lock = threading.Lock()  # lock used to ensure thread-safe (multi browsers/tabs)
app = Flask(__name__)
#app.run(host="0.0.0.0", port="8000", debug=True, threaded=True, use_reloader=False)
@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# Data collector
print("Setup DATA Collector")
free_dir = 'dataset/free'
blocked_dir = 'dataset/blocked'
try:
    os.makedirs(free_dir)
    os.makedirs(blocked_dir)
except FileExistsError:
    print('Directories not created because they already exist')
free_count = len(os.listdir(free_dir))
blocked_count = len(os.listdir(blocked_dir))
print("Free    Count : ", free_count)
print("Blocked Count : ", blocked_count)

# Camera
width = 300
height = 300
camera = Camera(width=width, height=height, rotate=False)

# SSD Object detector
print("Loading SSD Object Detector")
model = ObjectDetector('models/object_detect/ssd_mobilenet_v2_coco.engine')

# Collision Detector
print("Loading Collision Model")
collision_model = torchvision.models.alexnet(pretrained=False)
collision_model.classifier[6] = torch.nn.Linear(collision_model.classifier[6].in_features, 2)
collision_model.load_state_dict(torch.load('models/classification/best_model.pth'))
device = torch.device('cuda')
collision_model = collision_model.to(device)
mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])
normalize = torchvision.transforms.Normalize(mean, stdev)
check_time_blocked = time.time()
check_time_free = time.time()

# Robot Movement
output_pins = {
        'JETSON_XAVIER': 18,
        'JETSON_NANO': 33,
      }
output_pin = output_pins.get(GPIO.model, None)
if output_pin is None:
    raise Exception('PWM not supported on this board')

# Pin Setup:
# Board pin-numbering scheme
#GPIO.setmode(GPIO.BOARD)
# set pin as an output pin with optional initial state of HIGH
#GPIO.setup(output_pin, GPIO.OUT)

#duty_cycle = 10 # 2.5 = 0deg to 12.5=180deg
#freq = 50 # servo motor expect a pulse every 20ms (period), that means 50 pulses per second or Hertz
#p = GPIO.PWM(output_pin, freq)
#p.start(duty_cycle)
#time.sleep(1)
#p.ChangeDutyCycle(10)  # turn towards 90 degree
#time.sleep(1) # sleep 1 second
#p.ChangeDutyCycle(20)  # turn towards 0 degree
#time.sleep(1) # sleep 1 second
#p.ChangeDutyCycle(30) # turn towards 180 degree
#time.sleep(1) # sleep 1 second 
#pitch_pwm = pitch_angle*(2000/180) + 0   # pwm = ang(sweep/180) +/- offset
#pi.set_servo_pulsewidth(pitch_servo, pitch_pwm)

# Buffer for GCODE serial stream
RX_BUFFER_SIZE = 128

verbose = True
l_count = 0
g_count = 0
c_line = []
j_x = 0
j_y = 0
g_busy = False
g_stop = True

# Open grbl serial port
print("Setup serial port")
s = serial.Serial('/dev/ttyACM0', 115200)

# Wake up grbl
s.write(b"\r\n\r\n")
time.sleep(2)   # Wait for grbl to initialize
s.flushInput()  # Flush startup text in serial input

# CNC acceleration setttings
print("Setup CNC controller")
l_block = bytes("$120=100", 'utf-8')
s.write(l_block + b'\n') # Send g-code block to grbl
time.sleep(1)
l_block = bytes("$121=100", 'utf-8')
s.write(l_block + b'\n') # Send g-code block to grbl
time.sleep(1)


# Execute MAIN
if __name__ == '__main__':
    main()

