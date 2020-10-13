import traitlets
#from traitlets.config.configurable import SingletonConfigurable
import atexit
import cv2
import threading
import numpy as np


class Camera:
    def __init__(self, width, height, rotate, *args, **kwargs):
        # config
        self.width = width
        self.height = height
        self.fps = 21
        self.capture_width = 1280
        self.capture_height = 720
        self.rotate = rotate
        # Use Below in case the camera is mounted normally
        self.gst_source = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, ' \
                          'format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv ! video/x-raw, ' \
                          'width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink'\
                          % (self.capture_width, self.capture_height, self.fps, self.width, self.height)
        if self.rotate:
            # Use Below to Rotate Video 180deg in case the camera is mounted upside down
            self.gst_source = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, ' \
                              'format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv ! video/x-raw, ' \
                              'width=(int)%d, height=(int)%d, ' \
                              'format=(string)BGRx ! videoflip method=rotate-180 ! videoconvert ! appsink' \
                              % (self.capture_width, self.capture_height, self.fps, self.width, self.height)

        self.image_array = np.empty((self.height, self.width, 3), dtype=np.uint8)


        try:
            self.cap = cv2.VideoCapture(self.gst_source, cv2.CAP_GSTREAMER)

            re, img = self.cap.read()

            if not re:
                raise RuntimeError('camera capture error')

            self.image_array = img
            self.start()
        except:
            self.stop()
            raise RuntimeError('could not initialize camera')

        atexit.register(self.stop)


    def capture_frames(self):
        while True:
            re, img = self.cap.read()
            if re:
                if self.rotate:
                    self.image_array = cv2.rotate(img, cv2.ROTATE_180)
                else:
                    self.image_array = img
            else:
                break


    def exec_rotate(self):
        self.rotate = not(self.rotate)

    
    def start(self):
        if not self.cap.isOpened():
            self.cap.open(self.gst_source, cv2.CAP_GSTREAMER)
        if not hasattr(self, 'thread') or not self.thread.isAlive():
            self.thread = threading.Thread(target=self.capture_frames)
            self.thread.start()


    def stop(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'thread'):
            self.thread.join()
        del self.cap
            
            
    def restart(self):
        self.stop()
        self.start()
