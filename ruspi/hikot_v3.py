# USAGE
# python hikot.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from __future__ import print_function
import os,threading,time
from ctypes import *
from datetime import datetime
from threading import Thread

#from PIL import ImageFont, ImageDraw, Image
from collections import deque
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2


__author__ = 'J!mmyNeuTRON'

####################################################
### Parse arguments & Setup Object to track      ###
####################################################

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
print("[INFO] model loaded, starting program...")

counter = 0
(dX, dY) = (0, 0)
direction = ""

####################################################
### Load the .so                                 ###
####################################################
#print(os.environ)
HC = cdll.LoadLibrary('/home/nvidia/projects/frameworks/Hikvision/lib/libhcnetsdk.so.1')

####################################################
### Global variables                             ###
####################################################
SERIALNO_LEN = 48
STREAM_ID_LEN = 32
__curdir__ = os.getcwd()
__capdir__ = os.path.join(__curdir__,"capture")
__logdir__ = os.path.join(__curdir__,"logs")
exitFlag = 0

####################################################
### Data types to be used in the library         ###
####################################################
BOOL = ctypes.c_bool
INT = ctypes.c_int
LONG = ctypes.c_long
BYTE = ctypes.c_ubyte
WORD = ctypes.c_ushort
DWORD = ctypes.c_uint
CHARP = ctypes.c_char_p
VOIDP = ctypes.c_void_p
HWND = ctypes.c_uint

CMPFUNC = ctypes.CFUNCTYPE(None,LONG,DWORD,BYTE,DWORD,VOIDP)

####################################################
### Structures to be used in the library         ###
####################################################
class NET_DVR_DEVICEINFO_V30(ctypes.Structure):
    _fields_ = [("sSerialNumber",BYTE * SERIALNO_LEN),
                ("byAlarmInPortNum",BYTE),
                ("byAlarmOutPortNum",BYTE),
                ("byDiskNum",BYTE),
                ("byDVRType",BYTE),
                ("byChanNum",BYTE),
                ("byStartChan",BYTE),
                ("byAudioChanNum",BYTE),
                ("byIPChanNum",BYTE),
                ("byZeroChanNum",BYTE),
                ("byMainProto",BYTE),
                ("bySubProto",BYTE),
                ("bySupport",BYTE),
                ("bySupport1",BYTE),
                ("bySupport2",BYTE),
                ("wDevType",WORD),
                ("bySupport3",BYTE),
                ("byMultiStreamProto",BYTE),
                ("byStartDChan",BYTE),
                ("byStartDTalkChan",BYTE),
                ("bySupport4",BYTE),
                ("byLanguageType",BYTE),
                ("byRes2",BYTE * 10)
                ]
                #("byHighDChanNum",BYTE),
LPNET_DVR_DEVICEINFO_V30 = ctypes.POINTER(NET_DVR_DEVICEINFO_V30)

class NET_DVR_CLIENTINFO(ctypes.Structure):
    _fields_ = [
                ("lChannel",LONG),
                ("lLinkMode",LONG),
                ("hPlayWnd",HWND),
                ("sMultiCastIP",CHARP),
                ("byProtoType",BYTE),
                ("byRes",BYTE*3)
                ]
LPNET_DVR_CLIENTINFO = ctypes.POINTER(NET_DVR_CLIENTINFO)

class NET_DVR_PREVIEWINFO(ctypes.Structure):
    _fields_ = [
                ("lChannel",LONG),
                ("dwStreamType",DWORD),
                ("dwLinkMode",DWORD),
                ("hPlayWnd",HWND),
                ("bBlocked",DWORD),
                ("bPassbackRecord",DWORD),
                ("byPreviewMode",BYTE),
                ("byStreamID",BYTE * STREAM_ID_LEN),
                ("byProtoType",BYTE),
                ("byRes",BYTE * 222),
                ]
LPNET_DVR_PREVIEWINFO = ctypes.POINTER(NET_DVR_PREVIEWINFO)

class NET_DVR_SDKLOCAL_CFG(ctypes.Structure):
    _fields_ = [
                ("byEnableAbilityParse",BYTE),
                ("byVoiceComMode",BYTE),
                ("byRes",BYTE*382),
                ("byProtectKey",BYTE*128)
                ]
LPNET_DVR_SDKLOCAL_CFG = ctypes.POINTER(NET_DVR_SDKLOCAL_CFG)

class NET_DVR_JPEGPARA(ctypes.Structure):
    _fields_ = [
                ("wPicSize",WORD),
                ("wPicQuality",WORD)
                ]

LPNET_DVR_JPEGPARA = ctypes.POINTER(NET_DVR_JPEGPARA)

####################################################
### Error codes                                  ###
####################################################
__errorcodes__ = {
0: 'No error',
3: 'SDK is not initialized',
6: 'Version mismatch. SDK version is not matching with the device.',
7: 'Failed to connect to the device. The device is off-line, or connection timeout caused by network',
8: 'Failed to send data to the device.',
9: 'Failed to receive data from the device.',
10: 'Timeout when receiving the data from the device',
12: 'API calling order error',
17: 'Parameter error. Input or output parameter in the SDK API is NULL.',
34: 'Failed to create a file, during local recording, saving picture, getting configuration file or downloading record file',
41: 'Resource allocation error.',
43: 'Buffer to save stream or picture data is not enough.',
44: 'Establish SOCKET error.',
47: 'User doest not exist. The user ID has been logged out or unavailable.',
72: 'Failed to bind socket.',
73: 'Socket disconnected. It is caused by network disconnection or destination unreachable.',
76: 'SDK program exception.',
84: 'Load StreamTransClient.dll failed'
}

####################################################
### SDK Information functions                    ###
####################################################
def getSDKVersion():
    """
    Get SDK version information
    @Params
    None
    @Return
    SDK Version information [Call getLastError() to get the error code]
    """
    _gsv = HC.NET_DVR_GetSDKVersion
    _gsv.restype = DWORD
    return hex(_gsv())

####################################################
### SDK initialization and termination functions ###
####################################################
def init(filePtr):
    """
    Initialize the SDK. Call this function before using any of the other APIs
    @Params
    filePtr  - File pointer to the SDK log file
    @Return
    None
    """
    _init = HC.NET_DVR_Init
    _init.restype = BOOL
    if _init():
        print(str(datetime.now())+"|INFO|SDK initialized successfully",file=filePtr)
    else :
        _m=str(datetime.now())+"|ERROR|SDK initialization failed. Error message: "+getErrorMsg(getLastError())
        print(_m,file=filePtr)
        raise Exception(_m)

def release(filePtr):
    """
    Release the SDK. If init() was called, invoke this function at program exit
    @Params
    filePtr  - File pointer to the SDK log file
    @Return
    None
    """
    _release = HC.NET_DVR_Cleanup
    _release.restype = BOOL
    if _release():
        print(str(datetime.now())+"|INFO|SDK released successfully",file=filePtr)
    else :
        _m=str(datetime.now())+"|ERROR|SDK release failed. Error message: "+getErrorMsg(getLastError())
        print(_m,file=filePtr)
        raise Exception(_m)

####################################################
### Connection timeout functions                 ###
####################################################
def setConnectTime(timeout,attemptCount,filePtr):
    """
    Set network connection timeout and connection attempt times. Default timeout is 3s.
    @Params
    timeout - timeout, unit:ms, range:[300,75000]
    attemptCount - Number of attempts for connection
    filePtr  - File pointer to the SDK log file
    @Return
    None
    """
    _sct = HC.NET_DVR_SetConnectTime
    _sct.argtypes = [DWORD, DWORD]
    _sct.restype = BOOL
    if _sct(timeout,attemptCount):
        print(str(datetime.now())+"|INFO|Set connect time-"+str(timeout)+":"+str(attemptCount),file=filePtr)
    else:
        print(str(datetime.now())+"|ERROR|Set connect time failed",file=filePtr)

def setReconnect(interval,enableReconnect,filePtr):
    """
    Set reconnecting time interval. Default reconnect interval is 5 seconds.
    @Params
    interval - Reconnecting interval, unit:ms, default:30s
    enableReconnect - Enable or disable reconnect function, 0-disable, 1-enable(default)
    filePtr  - File pointer to the SDK log file
    @Return
    None
    """
    _sr = HC.NET_DVR_SetReconnect
    _sr.argtypes = [DWORD, DWORD]
    _sr.restype = BOOL
    if _sr(interval,enableReconnect):
        print(str(datetime.now())+"|INFO|Set reconnect time-"+str(interval)+":"+str(enableReconnect),file=filePtr)
    else:
        print(str(datetime.now())+"|ERROR|Set reconnect time failed",file=filePtr)

####################################################
### Error message functions                      ###
####################################################
def getLastError():
    """
    The error code of last operation
    @Params
    None
    @Return
    Error code
    """
    _gle = HC.NET_DVR_GetLastError
    _gle.restype = DWORD
    return _gle()

def getErrorMsg(errorCode):
    """
    Return the error message of last operation
    @Params
    errorCode - Error code from getLastError()
    @Return
    Error message
    """
    return __errorcodes__[errorCode]

####################################################
### Device login functions                       ###
####################################################

def login(dIP,dPort,username,password,filePtr):
    """
    Login to the device
    @Params
    dIP - IP address of the device
    dPort - Port number of the device
    username - Username for login
    password - password
    filePtr  - File pointer to the SDK log file
    @Return
    (userID, dInfo) - Unique user ID and device info, else -1 on failure [Call getLastError() to get the error code]
    """
    _info = NET_DVR_DEVICEINFO_V30()
    _userId = HC.NET_DVR_Login_V30(dIP,dPort,username,password,ctypes.byref(_info))

    if _userId != -1 :
        print(str(datetime.now())+"|INFO|Logged in successfully",file=filePtr)
        return _userId,_info
    else :
        _m = str(datetime.now())+"|INFO|Login failed. Error message: "+getErrorMsg(getLastError())
        print(_m,file=filePtr)
        raise Exception(_m)

def logout(userId,filePtr):
    """
    Logout from the device
    @Params
    userID - User ID, returned from login()
    filePtr  - File pointer to the SDK log file
    @Return
    None
    """
    _lo = HC.NET_DVR_Logout
    _lo.argtypes = [LONG]
    _lo.restype = BOOL
    _ldir = os.path.join(__logdir__,'SDK.log')
    f = open(_ldir,'a')
    if _lo(userId):
        print(str(datetime.now())+"|INFO|Logged out successfully",file=filePtr)
    else :
        _m = str(datetime.now())+"|ERROR|Logout failed. Error message: "+getErrorMsg(getLastError())
        print(_m,file=filePtr)
        raise Exception(_m)

####################################################
### RTSP functions                               ###
####################################################

def startRTSP_OT(userId,f):
    """
    Start RTSP stream Object Tracker for device in OpenCV
    @Params
    userID - User ID, returned from login()
    filePtr  - File pointer to the SDK log file
    @Return
    None
    """  

    # define ellipse sight regions
    global sight_x
    global sight_y
    sight_x = 640
    sight_y = 360
    # initialize object target regions
    global obj_tar_x
    global obj_tar_y
    obj_tar_x = 640
    obj_tar_y = 360
    global target_in_sight
    target_in_sight = 0
    # Number of explicitly defined detections
    global count
    count = 0
    # Test polygon to check in target is in this area
    global el_maj
    global el_min
    el_maj = 100
    el_min = 100

    # initialize the video stream, allow the cammera sensor to warmup,
    # and initialize the FPS counter
    print("[INFO] starting video stream...")
    #vs = VideoStream(src=0).start()
    vs = VideoStream("rtsp://admin:JesusisChrist#12@192.168.1.64:554/Streaming/Channels/102/").start()
    # vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)
    fps = FPS().start()    
    # Define center of the screen
    center = 640,360

    pts = deque(maxlen=args["buffer"])
    # Setup buffer for target xy average, initialise buffer with zeros
    # Adjust buffersize to determine speed of movement
    pts_ar = deque(maxlen=20)
    tracked_targets = 0,0,0
    pts_ar.append(tracked_targets)

    # Move PTZ to object target
    tPTZ = Thread(target=setPTZ, args=(userId,f))
    tPTZ.daemon = True
    tPTZ.start()
    
    font = ImageFont.truetype("font/Xolonium-Regular.ttf", 15) 
    label = ""
    startX, startY, endX, endY = 1,1,1,1

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        in_frame = vs.read()
        frame = cv2.resize(in_frame, (1280,720), interpolation = cv2.INTER_AREA)
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
                #label = "{}-[ {:.2f}%]".format(CLASSES[idx],confidence * 100)
                label = "Target[ {:.2f}%]".format(round(confidence * 100,2))

                if CLASSES[idx] == "person":
                    count = count + 1
                    center = (int(startX + (endX - startX)/2),int(startY + (endY - startY)*0.3))
                    tracked_targets = (count,int(startX + (endX - startX)/2),int(startY + (endY - startY)*0.3))
                    # update the tracking points queue
                    pts.appendleft(center)
                    pts_ar.append(tracked_targets)

                    #obj_tar_x = int(startX + (endX - startX)/2)
                    #obj_tar_y = int(startY + (endY - startY)*0.3)
                    """
                    if count == 10:
                        count = 0
                        
                        trig = True
                    else:
                        count = count + 1
                        trig = False
                    """
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

                    cv2.rectangle(frame,((int(startX + (endX - startX)/2))-30,(int(startY + (endY - startY)*0.3)-30))
                        ,((int(startX + (endX - startX)/2))+30,(int(startY + (endY - startY)*0.3)+30)),(0,0,255),1)                    				
                    #cv2.circle(frame,center,30,(0,0,255),1)
                    cv2.line(frame,ch1_1,ch1_2,(0,0,255),1)
                    cv2.line(frame,ch2_1,ch2_2,(0,0,255),1)
                    cv2.line(frame,ch3_1,ch3_2,(0,0,255),1)
                    cv2.line(frame,ch4_1,ch4_2,(0,0,255),1)

                    img_pil = Image.fromarray(frame)
                    draw = ImageDraw.Draw(img_pil)
                    tlabel = label + "-" + str(count)
                    draw.text((int((startX + (endX - startX)/2)-70), int((startY + (endY - startY)*0.3))-50),  tlabel, font = font, fill = (0, 0, 255, 0))
                    frame = np.array(img_pil)
                    #cv2.putText(frame, label + "-" + str(count), (int((startX + (endX - startX)/2)-70), int((startY + (endY - startY)*0.3))-50), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,0), 2)

                    # loop over the set of tracked points
			
                    """for i in range(1, len(pts)):
                        # if either of the tracked points are None, ignore them
                        if pts[i - 1] is None or pts[i] is None:
                            continue

                        # otherwise, compute the thickness of the line and
                        # draw the connecting lines
                        #thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 0.5)
                        cv2.circle(frame,pts[i],1,(255,255,0),1)"""

                # Draw sight based on average points of the target array
                
                if count is None or count == 0:
                    obj_tar_x = 640
                    obj_tar_y = 360
                    cv2.circle(frame,(640,360),1,(0,255,0),2)
                    #el_maj = 100
                    #el_min = 100
                else:
                    # Draw selected target tracked points
                    k = 0
                    Av_x = 0
                    Av_y = 0
                    for i in range(1, len(pts_ar)):
                        if pts_ar[i - 1] is None or pts_ar[i] is None:
                            continue
                        x,y = pts_ar[i][1],pts_ar[i][2]
                        cv2.circle(frame,(x,y),1,(0,255,0),1)
                        
                        if pts_ar[i][0] == 1:
                            #obj_tar_x = pts_ar[i][1]
                            #obj_tar_y = pts_ar[i][2]
                            k = k +1
                            Av_x += pts_ar[i][1]
                            Av_y += pts_ar[i][2]
                    if k > 0:
                        obj_tar_x = int(round(Av_x / k)) 
                        obj_tar_y = int(round(Av_y / k))
                        target_center = obj_tar_x,obj_tar_y
                        # check if target is in sight 
                        sight_ell = cv2.ellipse2Poly((sight_x,sight_y),(int(el_maj),int(el_min)),0,0,360,15)  
                        target_in_sight = cv2.pointPolygonTest(sight_ell,target_center,False)
                        if target_in_sight != -1:
                            if el_maj > 20:
                                el_maj -= 3
                            if el_min > 20:
                                el_min -= 3
                    cv2.circle(frame,(obj_tar_x,obj_tar_y),1,(0,0,255),2)
                    
                 
                    """arr = target_arr
                    obj_tar_x = int(np.around(arr.mean(0),decimals=0)[0:1])
                    obj_tar_y = int(np.around(arr.mean(0),decimals=0)[1:2])
                    cv2.circle(frame,(obj_tar_x,obj_tar_y),1,(0,0,255),2)"""

                    """arr = np.array(pts_ar)
                    target_num = int(np.around(arr.mean(0),decimals=0)[0:1])                
                    obj_tar_x = int(np.around(arr.mean(0),decimals=0)[1:2])
                    obj_tar_y = int(np.around(arr.mean(0),decimals=0)[2:3])
                    cv2.circle(frame,(obj_tar_x,obj_tar_y),1,(0,0,255),2)"""             

                # Add bounding boxes on the rest of the objects
                """
                cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)    
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)    
                """

        # draw ellipse target on screen
        cv2.ellipse(frame,(sight_x,sight_y),(el_maj,el_min),0,0,360,(0,255,255),1)
        #cv2.ellipse(frame,(sight_x,sight_y),(30,30),0,0,360,(0,100,255),1)
        #x-points
        ch1_1 = int(sight_x-100),int(sight_y)
        ch1_2 = int(sight_x-15),int(sight_y)
        ch2_1 = int(sight_x+100),int(sight_y)
        ch2_2 = int(sight_x+15),int(sight_y)

        ch1_3 = int(sight_x-5),int(sight_y-70)
        ch1_4 = int(sight_x+5),int(sight_y-70)
        ch2_3 = int(sight_x-5),int(sight_y+70)
        ch2_4 = int(sight_x+5),int(sight_y+70)
        #y-points
        ch3_1 = int(sight_x),int(sight_y-70)
        ch3_2 = int(sight_x),int(sight_y-15)
        ch4_1 = int(sight_x),int(sight_y+70)
        ch4_2 = int(sight_x),int(sight_y+15)

        ch3_3 = int(sight_x-100),int(sight_y-5)
        ch3_4 = int(sight_x-100),int(sight_y+5)
        ch4_3 = int(sight_x+100),int(sight_y-5)
        ch4_4 = int(sight_x+100),int(sight_y+5)
				                        				
        cv2.line(frame,ch1_1,ch1_2,(255,255,255),2)
        cv2.line(frame,ch2_1,ch2_2,(255,255,255),2)
        cv2.line(frame,ch1_3,ch1_4,(255,255,255),2)
        cv2.line(frame,ch2_3,ch2_4,(255,255,255),2)

        cv2.line(frame,ch3_1,ch3_2,(255,255,255),2)
        cv2.line(frame,ch4_1,ch4_2,(255,255,255),2)
        cv2.line(frame,ch3_3,ch3_4,(255,255,255),2)
        cv2.line(frame,ch4_3,ch4_4,(255,255,255),2)
        
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # if the `q` key was pressed, break from the loop
        if key == ord("w"):
            setPTZ_start_stop(userId,21,7,0.2,f)
            pass
        if key == ord("s"):
            setPTZ_start_stop(userId,22,7,0.2,f)
            pass
        if key == ord("a"):
            setPTZ_start_stop(userId,23,7,0.2,f)
            pass
        if key == ord("d"):
            setPTZ_start_stop(userId,24,7,0.2,f)
            pass
        if key == ord("m"):
            setPTZ_start_stop(userId,11,7,0.2,f)
            pass
        if key == ord("n"):
            setPTZ_start_stop(userId,12,7,0.2,f)
            pass
        if key == ord("p"):
            setPTZ_start_stop(userId,29,3,1,f)
            pass
        if key == ord("q"):
            setPTZleft_stop(userId,23,3,f)
            setPTZright_stop(userId,24,3,f)
            setPTZup_stop(userId,21,3,f)
            setPTZdown_stop(userId,22,3,f)
            break

        # update the FPS counter
        fps.update()
        count = 0

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

####################################################
### PTZ functions                                ###
####################################################
def setPTZ(userId,f):
    """
    Control PTZ for device
    @Params
    userID - User ID, returned from login()
    filePtr  - File pointer to the SDK log file
    @Return
    None
    """
    print("[INFO] PTZ started")
    global target_in_sight
    global obj_tar_x
    global obj_tar_y
    global el_maj
    global el_min
    
    while (1):
        if target_in_sight == -1:
            # Target left top of the sight
            if (obj_tar_x < sight_x) and (obj_tar_y < sight_y):
                speed = 2
                delay = 0.15
                if (sight_x - obj_tar_x) >= 640 and (sight_y - obj_tar_y) >= 640:
                    speed = 6
                    delay = 0.5 
                elif (sight_x - obj_tar_x) >= 317 and (sight_y - obj_tar_y) >= 317:
                    speed = 6
                    delay = 0.2
                elif (sight_x - obj_tar_x) >= 158 and (sight_y - obj_tar_y) >= 158:
                    speed = 4
                    delay = 0.2              
                elif (sight_x - obj_tar_x) >= 79 and (sight_y - obj_tar_y) >= 79:
                    speed = 2
                    delay = 0.2 
                elif (sight_x - obj_tar_x) >= 20 and (sight_y - obj_tar_y) >= 20:
                    el_maj = 100
                    el_min = 100       
                else:
                    speed = 2
                    delay = 0.15  
                setPTZ_start_stop(userId,25,speed,delay,f)  
            # Target right top of the sight
            elif (obj_tar_x > sight_x) and (obj_tar_y < sight_y):
                speed = 2
                delay = 0.15
                if (obj_tar_x - sight_x) >= 640 and (sight_y - obj_tar_y) >= 640:
                    speed = 6
                    delay = 0.5 
                elif (obj_tar_x - sight_x) >= 317 and (sight_y - obj_tar_y) >= 317:
                    speed = 6
                    delay = 0.2
                elif (obj_tar_x - sight_x) >= 158 and (sight_y - obj_tar_y) >= 158:
                    speed = 4
                    delay = 0.2              
                elif (obj_tar_x - sight_x) >= 79 and (sight_y - obj_tar_y) >= 79:
                    speed = 2
                    delay = 0.2 
                elif (obj_tar_x - sight_x) >= 20 and (sight_y - obj_tar_y) >= 20:
                    el_maj = 100
                    el_min = 100           
                else:
                    speed = 2
                    delay = 0.15  
                setPTZ_start_stop(userId,26,speed,delay,f) 
            # Target left bottom of the sight
            elif (obj_tar_x < sight_x) and (obj_tar_y > sight_y):
                speed = 2
                delay = 0.15
                if (sight_x - obj_tar_x) >= 640 and (obj_tar_y - sight_y) >= 640:
                    speed = 6
                    delay = 0.5 
                elif (sight_x - obj_tar_x) >= 317 and (obj_tar_y - sight_y) >= 317:
                    speed = 6
                    delay = 0.2
                elif (sight_x - obj_tar_x) >= 158 and (obj_tar_y - sight_y) >= 158:
                    speed = 4
                    delay = 0.2              
                elif (sight_x - obj_tar_x) >= 79 and (obj_tar_y - sight_y) >= 79:
                    speed = 2
                    delay = 0.2  
                elif (sight_x - obj_tar_x) >= 20 and (obj_tar_y - sight_y) >= 20:
                    el_maj = 100
                    el_min = 100      
                else:
                    speed = 2
                    delay = 0.15  
                setPTZ_start_stop(userId,27,speed,delay,f) 
            # Target right bottom of the sight
            elif (obj_tar_x > sight_x) and (obj_tar_y > sight_y):
                speed = 2
                delay = 0.15
                if (obj_tar_x - sight_x) >= 640 and (obj_tar_y - sight_y) >= 640:
                    speed = 6
                    delay = 0.5 
                elif (obj_tar_x - sight_x) >= 317 and (obj_tar_y - sight_y) >= 317:
                    speed = 6
                    delay = 0.2
                elif (obj_tar_x - sight_x) >= 158 and (obj_tar_y - sight_y) >= 158:
                    speed = 4
                    delay = 0.2              
                elif (obj_tar_x - sight_x) >= 79 and (obj_tar_y - sight_y) >= 79:
                    speed = 2
                    delay = 0.2  
                elif (obj_tar_x - sight_x) >= 20 and (obj_tar_y - sight_y) >= 20:
                    el_maj = 100
                    el_min = 100      
                else:
                    speed = 2
                    delay = 0.15  
                setPTZ_start_stop(userId,28,speed,delay,f) 

            # Target left of the sight
            if obj_tar_x < sight_x: 
                speed = 2
                delay = 0.15 
                if (sight_x - obj_tar_x) >= 1250:
                    speed = 7
                    delay = 0.2
                elif (sight_x - obj_tar_x) >= 640:
                    speed = 6
                    delay = 0.5  
                elif (sight_x - obj_tar_x) >= 317:
                    speed = 6
                    delay = 0.2
                elif (sight_x - obj_tar_x) >= 158:
                    speed = 4
                    delay = 0.2              
                elif (sight_x - obj_tar_x) >= 79:
                    speed = 2
                    delay = 0.2 
                elif (sight_x - obj_tar_x) >= 20:
                    el_maj = 100
                    el_min = 100       
                else:
                    speed = 2
                    delay = 0.15  
                setPTZ_start_stop(userId,23,speed,delay,f)

			# Target right of the sight	    	
            elif obj_tar_x > sight_x:
                speed = 2
                delay = 0.15
                if (obj_tar_x - sight_x) >= 1250:
                    speed = 7
                    delay = 0.2
                elif (obj_tar_x - sight_x) >= 640:
                    speed = 6
                    delay = 0.5  
                elif (obj_tar_x - sight_x) >= 317:
                    speed = 6
                    delay = 0.2
                elif (obj_tar_x - sight_x) >= 158:
                    speed = 4
                    delay = 0.2              
                elif (obj_tar_x - sight_x) >= 79:
                    speed = 2
                    delay = 0.2 
                elif (obj_tar_x - sight_x) >= 20:
                    el_maj = 100
                    el_min = 100       
                else:
                    speed = 2
                    delay = 0.15    
                setPTZ_start_stop(userId,24,speed,delay,f)
			    		
            # Target top of the sight        
            if obj_tar_y < sight_y:
                speed = 2
                delay = 0.15
                if (sight_y - obj_tar_y) >= 700:
                    speed = 6
                    delay = 0.2
                elif (sight_y - obj_tar_y) >= 360:
                    speed = 5
                    delay = 0.2  
                elif (sight_y - obj_tar_y) >= 180:
                    speed = 4
                    delay = 0.2
                elif (sight_y - obj_tar_y) >= 90:
                    speed = 2
                    delay = 0.6              
                elif (sight_y - obj_tar_y) >= 45:
                    speed = 2
                    delay = 0.3 
                elif (sight_y - obj_tar_y) >= 20:
                    el_maj = 100
                    el_min = 100        
                else:
                    speed = 2
                    delay = 0.15    
                setPTZ_start_stop(userId,21,speed,delay,f)

            # Target bottom of the sight
            elif obj_tar_y > sight_y:
                speed = 2
                delay = 0.15
                if (obj_tar_y - sight_y) >= 700:
                    speed = 6
                    delay = 0.2
                elif (obj_tar_y - sight_y) >= 360:
                    speed = 5
                    delay = 0.2  
                elif (obj_tar_y - sight_y) >= 180:
                    speed = 4
                    delay = 0.2
                elif (obj_tar_y - sight_y) >= 90:
                    speed = 2
                    delay = 0.6              
                elif (obj_tar_y - sight_y) >= 45:
                    speed = 2
                    delay = 0.3
                elif (obj_tar_y - sight_y) >= 20:
                    el_maj = 100
                    el_min = 100        
                else:
                    speed = 2
                    delay = 0.15        
                setPTZ_start_stop(userId,22,speed,delay,f)

def setPTZ_start_stop(userId,cmd,speed,delay,filePtr):
    """
    Control PTZ pan left for device
    @Params
    userID - User ID, returned from login()
    filePtr  - File pointer to the SDK log file
    @Return
    None
    """
    ClientInfo = NET_DVR_CLIENTINFO()
    ClientInfo.hPlayWnd = 0
    ClientInfo.lChannel = 1
    ClientInfo.lLinkMode    = 0
    ClientInfo.sMultiCastIP = None
    lRealPlayHandle = HC.NET_DVR_RealPlay_V30(userId, ctypes.byref(ClientInfo), None, None, 0)

    if lRealPlayHandle  < 0:
        print("pyd1---NET_DVR_RealPlay_V30 error\n")
    else:
        check = HC.NET_DVR_PTZControlWithSpeed(lRealPlayHandle,cmd,0,speed)
        time.sleep(delay)
        check = HC.NET_DVR_PTZControlWithSpeed(lRealPlayHandle,cmd,1,speed)
        if check:
            print(str(datetime.now())+"|INFO|PTZ Command executed ",file=filePtr)
        else:
            _m = str(datetime.now())+"|ERROR|PTZ Command failed. Error message: "+getErrorMsg(getLastError())
            print(_m,file=filePtr)
            raise Exception(_m) 

def setPTZleft_start(userId,cmd,speed,delay,filePtr):
    """
    Control PTZ pan left for device
    @Params
    userID - User ID, returned from login()
    filePtr  - File pointer to the SDK log file
    @Return
    None
    """
    ClientInfo = NET_DVR_CLIENTINFO()
    ClientInfo.hPlayWnd = 0
    ClientInfo.lChannel = 1
    ClientInfo.lLinkMode    = 0
    ClientInfo.sMultiCastIP = None
    lRealPlayHandle = HC.NET_DVR_RealPlay_V30(userId, ctypes.byref(ClientInfo), None, None, 0)

    if lRealPlayHandle  < 0:
        print("pyd1---NET_DVR_RealPlay_V30 error\n")
    else:
        check = HC.NET_DVR_PTZControlWithSpeed(lRealPlayHandle,cmd,0,speed)
        time.sleep(delay)
        check = HC.NET_DVR_PTZControlWithSpeed(lRealPlayHandle,cmd,1,speed)
        if check:
            print(str(datetime.now())+"|INFO|PTZ pan left",file=filePtr)
        else:
            _m = str(datetime.now())+"|ERROR|PTZ pan left failed. Error message: "+getErrorMsg(getLastError())
            print(_m,file=filePtr)
            raise Exception(_m) 

def setPTZright_start(userId,cmd,speed,delay,filePtr):
    """
    Control PTZ pan right for device
    @Params
    userID - User ID, returned from login()
    filePtr  - File pointer to the SDK log file
    @Return
    None
    """
    ClientInfo = NET_DVR_CLIENTINFO()
    ClientInfo.hPlayWnd = 0
    ClientInfo.lChannel = 1
    ClientInfo.lLinkMode    = 0
    ClientInfo.sMultiCastIP = None
    lRealPlayHandle = HC.NET_DVR_RealPlay_V30(userId, ctypes.byref(ClientInfo), None, None, 0)

    if lRealPlayHandle  < 0:
        print("pyd1---NET_DVR_RealPlay_V30 error\n")
    else:
        check = HC.NET_DVR_PTZControlWithSpeed(lRealPlayHandle,cmd,0,speed)
        time.sleep(delay)
        check = HC.NET_DVR_PTZControlWithSpeed(lRealPlayHandle,cmd,1,speed)
        if check:
            print(str(datetime.now())+"|INFO|PTZ pan right",file=filePtr)
        else:
            _m = str(datetime.now())+"|ERROR|PTZ pan right. Error message: "+getErrorMsg(getLastError())
            print(_m,file=filePtr)
            raise Exception(_m)

def setPTZup_start(userId,cmd,speed,delay,filePtr):
    """
    Control PTZ tilt up for device
    @Params
    userID - User ID, returned from login()
    filePtr  - File pointer to the SDK log file
    @Return
    None
    """
    ClientInfo = NET_DVR_CLIENTINFO()
    ClientInfo.hPlayWnd = 0
    ClientInfo.lChannel = 1
    ClientInfo.lLinkMode    = 0
    ClientInfo.sMultiCastIP = None
    lRealPlayHandle = HC.NET_DVR_RealPlay_V30(userId, ctypes.byref(ClientInfo), None, None, 0)

    if lRealPlayHandle  < 0:
        print("pyd1---NET_DVR_RealPlay_V30 error\n")
    else:
            check = HC.NET_DVR_PTZControlWithSpeed(lRealPlayHandle,cmd,0,speed)
            time.sleep(delay)
            check = HC.NET_DVR_PTZControlWithSpeed(lRealPlayHandle,cmd,1,speed)
            if check:
                print(str(datetime.now())+"|INFO|PTZ tilt up",file=filePtr)
            else:
                _m = str(datetime.now())+"|ERROR|PTZ tilt up. Error message: "+getErrorMsg(getLastError())
                print(_m,file=filePtr)
                raise Exception(_m)

def setPTZdown_start(userId,cmd,speed,delay,filePtr):
    """
    Control PTZ tilt down for device
    @Params
    userID - User ID, returned from login()
    filePtr  - File pointer to the SDK log file
    @Return
    None
    """
    ClientInfo = NET_DVR_CLIENTINFO()
    ClientInfo.hPlayWnd = 0
    ClientInfo.lChannel = 1
    ClientInfo.lLinkMode    = 0
    ClientInfo.sMultiCastIP = None
    lRealPlayHandle = HC.NET_DVR_RealPlay_V30(userId, ctypes.byref(ClientInfo), None, None, 0)

    if lRealPlayHandle  < 0:
        print("pyd1---NET_DVR_RealPlay_V30 error\n")
    else:
            check = HC.NET_DVR_PTZControlWithSpeed(lRealPlayHandle,cmd,0,speed)
            time.sleep(delay)
            check = HC.NET_DVR_PTZControlWithSpeed(lRealPlayHandle,cmd,1,speed)
            if check:
                print(str(datetime.now())+"|INFO|PTZ tilt down",file=filePtr)
            else:
                _m = str(datetime.now())+"|ERROR|PTZ tilt down. Error message: "+getErrorMsg(getLastError())
                print(_m,file=filePtr)
                raise Exception(_m)  

# PTZ Stop

def setPTZleft_stop(userId,cmd,speed,filePtr):
    """
    Control PTZ pan left stop for device
    @Params
    userID - User ID, returned from login()
    filePtr  - File pointer to the SDK log file
    @Return
    None
    """
    ClientInfo = NET_DVR_CLIENTINFO()
    ClientInfo.hPlayWnd = 0
    ClientInfo.lChannel = 1
    ClientInfo.lLinkMode    = 0
    ClientInfo.sMultiCastIP = None
    lRealPlayHandle = HC.NET_DVR_RealPlay_V30(userId, ctypes.byref(ClientInfo), None, None, 0)
    
    if lRealPlayHandle  < 0:
        print("pyd1---NET_DVR_RealPlay_V30 error\n")
    else:
        print(cmd)
        if cmd == 23:
            check = HC.NET_DVR_PTZControlWithSpeed(lRealPlayHandle,cmd,1,speed)
            if check:
                print(str(datetime.now())+"|INFO|PTZ pan left stop!",file=filePtr)
            else:
                _m = str(datetime.now())+"|ERROR|PTZ pan left failed. Error message: "+getErrorMsg(getLastError())
                print(_m,file=filePtr)
                raise Exception(_m)  

def setPTZright_stop(userId,cmd,speed,filePtr):
    """
    Control PTZ pan right stop for device
    @Params
    userID - User ID, returned from login()
    filePtr  - File pointer to the SDK log file
    @Return
    None
    """
    ClientInfo = NET_DVR_CLIENTINFO()
    ClientInfo.hPlayWnd = 0
    ClientInfo.lChannel = 1
    ClientInfo.lLinkMode    = 0
    ClientInfo.sMultiCastIP = None
    lRealPlayHandle = HC.NET_DVR_RealPlay_V30(userId, ctypes.byref(ClientInfo), None, None, 0)
    
    if lRealPlayHandle  < 0:
        print("pyd1---NET_DVR_RealPlay_V30 error\n")
    else:
        print(cmd)
        if cmd == 24:            
            check = HC.NET_DVR_PTZControlWithSpeed(lRealPlayHandle,cmd,1,speed)
            if check:
                print(str(datetime.now())+"|INFO|PTZ pan right stop!",file=filePtr)
            else:
                _m = str(datetime.now())+"|ERROR|PTZ pan right failed. Error message: "+getErrorMsg(getLastError())
                print(_m,file=filePtr)
                raise Exception(_m)   

def setPTZup_stop(userId,cmd,speed,filePtr):
    """
    Control PTZ tilt up stop for device
    @Params
    userID - User ID, returned from login()
    filePtr  - File pointer to the SDK log file
    @Return
    None
    """
    ClientInfo = NET_DVR_CLIENTINFO()
    ClientInfo.hPlayWnd = 0
    ClientInfo.lChannel = 1
    ClientInfo.lLinkMode    = 0
    ClientInfo.sMultiCastIP = None
    lRealPlayHandle = HC.NET_DVR_RealPlay_V30(userId, ctypes.byref(ClientInfo), None, None, 0)
    
    if lRealPlayHandle  < 0:
        print("pyd1---NET_DVR_Rea# Move PTZ to object target\n")
    else:
        print(cmd)
        if cmd == 21:
            check = HC.NET_DVR_PTZControlWithSpeed(lRealPlayHandle,cmd,1,speed)
            if check:
                print(str(datetime.now())+"|INFO|PTZ tilt up stop!",file=filePtr)
            else:
                _m = str(datetime.now())+"|ERROR|PTZ tilt up failed. Error message: "+getErrorMsg(getLastError())
                print(_m,file=filePtr)
                raise Exception(_m)   

def setPTZdown_stop(userId,cmd,speed,filePtr):
    """
    Control PTZ tilt down stop for device
    @Params
    userID - User ID, returned from login()
    filePtr  - File pointer to the SDK log file
    @Return
    None
    """
    ClientInfo = NET_DVR_CLIENTINFO()
    ClientInfo.hPlayWnd = 0
    ClientInfo.lChannel = 1
    ClientInfo.lLinkMode    = 0
    ClientInfo.sMultiCastIP = None
    lRealPlayHandle = HC.NET_DVR_RealPlay_V30(userId, ctypes.byref(ClientInfo), None, None, 0)
    
    if lRealPlayHandle  < 0:
        print("pyd1---NET_DVR_RealPlay_V30 error\n")
    else:
        print(cmd)
        if cmd == 22:
            check = HC.NET_DVR_PTZControlWithSpeed(lRealPlayHandle,cmd,1,speed)
            if check:
                print(str(datetime.now())+"|INFO|PTZ tilt down stop!",file=filePtr)
            else:
                _m = str(datetime.now())+"|ERROR|PTZ tilt down failed. Error message: "+getErrorMsg(getLastError())
                print(_m,file=filePtr)
                raise Exception(_m)    
       
####################################################
### Live view functions                          ###
####################################################

def startRealPlay(userId,ipClientInfo,realDataCbk,userData,blocked):
    """
    Starting live view
    @Params
    userId - return value of login()
    ipClientInfo - Live view parameter
    realDataCb - Real-time stream data callback function
    userData - User data
    blocked - Whether to set stream data requesting process blocked or not: 0-no, 1-yes
    @Return
    -1 on failure [Call getLastError() to get the error code]
    Other values - live view handle for use in stopRealPlay()
    """
    _srp = HC.NET_DVR_RealPlay_V30
    if realDataCbk:
        _srp.argtypes = [LONG,LPNET_DVR_CLIENTINFO,CMPFUNC,VOIDP,BOOL]
    else:
        _srp.argtypes = [LONG,LPNET_DVR_CLIENTINFO,VOIDP,VOIDP,BOOL]
    _srp.restype = LONG
    return _srp(userId,ctypes.byref(ipClientInfo),realDataCbk,userData,blocked)

def stopRealPlay(realHandle):
    """
    Stopping live view
    @Params
    realHandle - live view handle, return value from startRealPlay()
    @Return
    TRUE on success
    FALSE on failure [Call getLastError() to get the error code]
    """
    _strp = HC.NET_DVR_StopRealPlay
    _strp.argtypes = [LONG]
    _strp.restype = BOOL
    return _strp(realHandle)

def getRealPlayerIndex(realHandle):
    """
    Get player handle to use with other player SDK functions
    @Params
    realHandle - Live view handle, return value from startRealPlay()
    @Return
    -1 on failure [Call getLastError() to get the error code]
    Other values - live view handle
    """
    _grpi = HC.NET_DVR_GetRealPlayerIndex
    _grpi.argtypes = [LONG]
    _grpi.restype = INT
    return _grpi(realHandle)

####################################################
### Capture picture functions                    ###
####################################################

def captureJPEGPicture(userId,channelNo,jpegParam,fileName,filePtr):
    """
    Capture a frame and save to file
    @Params
    userId - User Id, return value from login()
    channelNo - Channel index for capturing the picture
    jpegParam - Target JPEG picture parameters
    fileName - URL to save picture
    filePtr - File pointer to the logfile
    @Return
    TRUE on success
    FALSE on failure [Call getLastError() to get the error code]
    """
    _cjp = HC.NET_DVR_CaptureJPEGPicture
    _cjp.argtypes = [LONG,LONG,LPNET_DVR_JPEGPARA,CHARP]
    _cjp.restype = BOOL
    if _cjp(userId,channelNo,ctypes.byref(jpegParam),fileName):
        print(str(datetime.now())+"|INFO|Picture captured successfully at "+fileName,file=filePtr)
    else:
        print(str(datetime.now())+"|ERROR|Picture capture failed. Error message: "+getErrorMsg(getLastError()),file=filePtr)


####################################################
### Callback functions                           ###
####################################################
def setRealDataCallBack(lRealHandle,cbRealDataCbk,dwUser):
    """
    Set callback function
    @Params
    lRealHandle - live view handle, return value from startRealPlay()
    cbRealDataCbk - Callback function
    dwUser - User data
    @Return
    TRUE on success
    FALSE on failure [Call getLastError() to get the error code]
    """
    _srdcb = HC.NET_DVR_SetRealDataCallBack
    _srdcb.argtypes = [LONG,CMPFUNC,DWORD]
    _srdcb.restype = BOOL
    return _srdcb(lRealHandle,cbRealDataCbk,dwUser)

####################################################
### Helper functions                             ###
####################################################
def struct2tuple(struct):
    """
    Convert a structure to a tuple
    @Params
    struct - ctypes structure object
    @Return
    Tuple containing the values of all the fields in the struct
    """
    _sf = NET_DVR_DEVICEINFO_V30._fields_
    _dict = {}
    for _fn,_ft in _sf:
        _v = struct.__getattribute__(_fn)
        if(type(_v)) != int:
            _v = ctypes.cast(_v,ctypes.c_char_p).value
        _dict[_fn] = _v
    return _dict

def logger():
    """
    Logger utility
    @Params
    None
    @Return
    None
    """
    if not os.path.exists(__logdir__):
        os.makedirs(__logdir__)
    _ldir = os.path.join(__logdir__,'SDK.log')
    f = open(_ldir,'w')
    print(str(datetime.now())+"|INFO|"+_ldir+" created",file=f)

def createDirectory(startName,count):
    """
    Creates a directory, if not exists
    @Params
    startName - starting name for the directory [numbers]
    count - count of the directories to be created
    @Return
    None
    """
    for _chan in  range(startName,count):
        _cdir = os.path.join(__capdir__,str(_chan))
        _ldir = os.path.join(__logdir__,str(_chan)+'chan.log')
        if not os.path.exists(_cdir):
            os.makedirs(_cdir)
        f = open(_ldir,'w')
        print(str(datetime.now())+"|INFO|"+_ldir+" created",file=f)

def checkVideoStatus(userId,channelNo,jpegParam):
    """
    Checks video status for the particular device, channel number and logs the information
    @Params
    userId - User Id, return value from login()
    channelNo - Channel index for capturing the picture
    jpegParam - Target JPEG picture parameters
    """
    global exitFlag
    _loop = 1
    _cdir = os.path.join(__capdir__,str(channelNo))
    _ldir = os.path.join(__logdir__,str(channelNo)+'chan.log')
    f=open(_ldir,'a')
    print(str(datetime.now())+"|INFO|Thread started for "+str(channelNo),file=f)

    while not exitFlag:
        print(str(datetime.now())+"|INFO|Thread called for "+str(channelNo),file=f)
        captureJPEGPicture(userId,channelNo,jpegParam,os.path.join(_cdir,str(channelNo)+'_'+str(_loop)+'.jpeg'),f)
        if _loop > 2:

            # Write code here for video status#

            os.remove(os.path.join(_cdir,str(channelNo)+'_'+str(_loop-2)+'.jpeg'))
        _loop+=1
        time.sleep(0.18)

    print(str(datetime.now())+"|INFO|Thread stopped for "+str(channelNo),file=f)

class CThread(threading.Thread):
    def __init__(self,threadId,name,userId,channelNo,jpegParam):
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.name = name
        self.userId = userId
        self.channelNo = channelNo
        self.jpegParam = jpegParam

    def run(self):
        checkVideoStatus(self.userId,self.channelNo,self.jpegParam)


####################################################
### Test functions                               ###
####################################################
def startCapture(ip,port,username,password,duration):
    """
    Start the capture for 16 video channels of the device and log the video status for every capture
    @Params
    ip - IP address of the device
    port - Port of the device
    username - Username for login
    password - Password for login
    duration - Duration for the test run
    @Return
    None
    """
    logger()
    f = open(os.path.join(__logdir__,'SDK.log'),'a')

    init(f)
    #init = HC.NET_DVR_Init()
    #if init:
    #    print("DVR Init success")
    #else:
    #    print("DVR Init failed")
    setConnectTime(300,7500,f)
    setReconnect(100,True,f)
    userId,deviceInfo = login(ip,port,username,password,f)
    if userId != -1:
        startRTSP_OT(userId,f)

    """
    dictDeviceInfo = struct2tuple(deviceInfo)
    startChan = dictDeviceInfo['byStartChan']
    chanNum = dictDeviceInfo['byChanNum']

    createDirectory(startChan,chanNum+1)

    jpegParam = NET_DVR_JPEGPARA()
    jpegParam.wPicSize = 2
    jpegParam.wPicQuality = 2

    for threadid in range(1,1):
        thread = CThread(threadid,'Channel'+str(threadid),userId,threadid,jpegParam)
        thread.start()


    startTime = int(time.time())
    endTime = duration * 3600
    print(str(datetime.now())+"|INFO|Startime:"+str(startTime)+";ETA in "+str(endTime)+"s",file=f)

    while True:
        if int(time.time()) - startTime > endTime:
            global exitFlag
            exitFlag = 1
            break
    """

    logout(userId,f)
    release(f)

if __name__ == "__main__":
    startCapture("192.168.1.64",8000,"admin","JesusisChrist#12",0.0066)
