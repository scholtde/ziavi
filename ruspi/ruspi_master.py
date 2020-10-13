# USAGE
# python targeting.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import keyboard
import pygame, sys
import pygame.locals
import xml.etree.ElementTree as ET     

# Functions
def prepareUI(flag):
  global windowSurface
  global Image
  global BACKG  	
  pygame.display.set_caption('Ruskubus v.1.0')
  if flag == 0:    
    WIDTH = 1024
    HEIGHT = 600
    windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN, 32)
  elif flag == 1:    
    WIDTH = 1024
    HEIGHT = 600
    windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
     
  #Icon = pygame.image.load('/home/nvidia/projects/ruspi/img/RCT_Logo_205x71l.png')
  BACKG = (0,0,0)
  x = 12 # (WIDTH * 0.90)
  y = 530 # (HEIGHT * 0.80)
  windowSurface.fill(BACKG)
  #windowSurface.blit(Icon, (x,y))
  #pygame.display.set_icon(Icon)
  pygame.display.update()
  pygame.display.flip()
  
  # Music/Sound effects
  #pygame.mixer.init()
  #pygame.mixer.music.load('/home/nvidia/projects/ruspi/audio/alert_1.wav')
  #pygame.mixer.music.play()
  #while pygame.mixer.music.get_busy() == True:
  #  continue

def writeText(path, text, xoffset, x, y, size, w, h, textCol, rectCol):
  switcher = {
    0: "/home/nvidia/projects/ruspi/fonts/LCARS.ttf",
    1: "/home/nvidia/projects/ruspi/fonts/LCARSGTJ2.ttf",
    2: "/home/nvidia/projects/ruspi/fonts/LCARSGTJ3.ttf",
  }
  #windowSurface.fill(BACKG) 
  #windowSurface.blit(Image, (x,y))  
  Rectangle = pygame.Rect(x-5, y, w, h*1.1)
  pygame.draw.rect(windowSurface,(rectCol),Rectangle,0) 
  font = pygame.font.Font(switcher.get(path, "/home/nvidia/projects/ruspi/fonts/LCARSGTJ2.ttf"), size)
  label = font.render(text, 1, (textCol))
  windowSurface.blit(label,(x+xoffset,y))
  #pygame.display.flip()  
  pygame.display.update(Rectangle)	
  return Rectangle

def Labels():
  """
  <GPS>
    <LAT>String</LAT>    0
    <LONG>String</LONG>  1
    <DATE>String</DATE>  2
    <TIME>String</TIME>  3
  </GPS>    
  """
  #(path, text, x, y, size, w, h, txtCol, recCol)
  #Draw sensor Labels
  y = 185
  x = 19
  size = 30
  w = 170
  offset = 43
  rec = 0, 77, 77
  txtCol = 255,255,255

  btnAL = writeText(1, "DATE", 125, x, y, size, w, size, txtCol, rec) 
  y = y+offset
  btnVT = writeText(1, "TIME", 125, x, y, size, w, size, txtCol, rec)  #Orange
  y = y+offset
  btnHT = writeText(1, "LATITUDE", 90, x, y, size, w, size, txtCol, rec) #Purple
  y = y+offset
  btnCT = writeText(1, "LONGITUDE", 78, x, y, size, w, size, txtCol, rec)
  
  
  """
  <IMU
    <ROLL>String</ROLL>    0
    <PITCH>String</PITCH>  1
    <YAW>String</YAW>  2
  </IMU>   
  """
  #(path, text, xoffset, x, y, size, w, h, textCol, recCol)
  rec = 0,0,0  
  writeText(1, "RUSPI",         0, 12, 5, 161, 290, 161, (255, 255, 255), rec)
  writeText(1, "RUSKUBUS v1.0", 0, 12, 150, 25, 150, 25, (29,136,133), rec) 
   
  #(path, text, x, y, size, w, h, txtCol, recCol)
  #Draw cycle counts
  x = 12
  y = 355
  size = 55
  w = 200
  offset = 60
  rec = 0,0,0
  txtCol = 255,255,255
  writeText(1, "IMU PITCH:", 25, x, y, size, w, size, txtCol, rec) 
  y = y+offset
  writeText(1, "IMU ROLL:", 38, x, y, size, w, size, txtCol, rec)  #Orange
  y = y+offset
  writeText(1, "IMU YAW:", 40, x, y, size, w, size, txtCol, rec) #Purple



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=False, default=str("MobileNetSSD_deploy.prototxt.txt"),
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=False, default=str("MobileNetSSD_deploy.caffemodel"),
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

pts = deque(maxlen=args["buffer"])

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")



stream = "udpsrc port=5700 ! application/x-rtp, media=video, clock-rate=90000, encoding-name=H264 ! rtph264depay ! avdec_h264 ! videoconvert ! appsink"
vs = VideoStream(stream).start()

time.sleep(2.0)
fps = FPS().start()
pygame.init()
# Load UI
prepareUI(0) 
Labels()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	in_frame = vs.read()
	frame = cv2.cvtColor(in_frame, cv2.COLOR_BGR2GRAY)
	frame = cv2.resize(in_frame, (600,380), interpolation = cv2.INTER_AREA)
	blob_frame = imutils.resize(in_frame, width=400)
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(blob_frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)

			if CLASSES[idx] == "person":
				center = (int(startX + (endX - startX)/2),int(startY + (endY - startY)*0.3))
				#x-points
				ch1_1 = (int((startX + (endX - startX)/2)-40),int(startY + (endY - startY)*0.3))
				ch1_2 = (int((startX + (endX - startX)/2)-20),int(startY + (endY - startY)*0.3))
				ch2_1 = (int((startX + (endX - startX)/2)+20),int(startY + (endY - startY)*0.3))
				ch2_2 = (int((startX + (endX - startX)/2)+40),int(startY + (endY - startY)*0.3))
				#y-points
				ch3_1 = (int(startX + (endX - startX)/2),int((startY + (endY - startY)*0.3)-40))
				ch3_2 = (int(startX + (endX - startX)/2),int((startY + (endY - startY)*0.3)-20))
				ch4_1 = (int(startX + (endX - startX)/2),int((startY + (endY - startY)*0.3)+20))
				ch4_2 = (int(startX + (endX - startX)/2),int((startY + (endY - startY)*0.3)+40))

				#cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)				
				cv2.circle(frame,center,30,(0,255,0),2)
				cv2.line(frame,ch1_1,ch1_2,(0,255,0),2)
				cv2.line(frame,ch2_1,ch2_2,(0,255,0),2)
				cv2.line(frame,ch3_1,ch3_2,(0,255,0),2)
				cv2.line(frame,ch4_1,ch4_2,(0,255,0),2)
				#y = startY - 15 if startY - 15 > 15 else startY + 15
				#cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
				cv2.putText(frame, label, (int((startX + (endX - startX)/2)-70), int((startY + (endY - startY)*0.3))-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
				
				# update the points queue
				pts.appendleft(center)


	
	# show the output frame
	#cv2.imshow("Frame", frame)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
	frame = pygame.surfarray.make_surface(frame)
#pygame.transform.flip(self.image, False, True), self.rect
	windowSurface.blit(pygame.transform.flip(frame, True, False), (300,5))
	Rectangle = pygame.Rect(300, 5, 600, 380)
	pygame.display.update(Rectangle)

	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	
	for event in pygame.event.get():
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_ESCAPE: 
				"""
				pygame.mixer.music.load('/home/nvidia/projects/ruspi/audio/deactivation_complete.wav')
				pygame.mixer.music.play()
				while pygame.mixer.music.get_busy() == True:
					continue
				"""
				cv2.destroyAllWindows()
				vs.stop()
				pygame.quit()
				sys.exit()
				quit() 				
				break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
pygame.quit()
sys.exit()
quit() 
