import os
import serial, time
import keyboard
import pygame, sys
import pygame.locals
import pigpio                                  
import time  
import xml.etree.ElementTree as ET     

pi = pigpio.pi() # Connect to local Pi.
throttle_gpio = 21
steering_gpio = 20
pitch_gpio = 16
roll_gpio = 12
pitch_servo = 16
roll_servo = 12
pi.set_servo_pulsewidth(throttle_gpio, 1500)
pi.set_servo_pulsewidth(steering_gpio, 1500)
pi.set_servo_pulsewidth(pitch_gpio, 1000)
pi.set_servo_pulsewidth(roll_gpio, 1500)
                        

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
    WIDTH = 500
    HEIGHT = 500
    windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
     
  Icon = pygame.image.load('/home/pi/projects/ruspi/main/img/RCT_Logo_205x71l.png')
  BACKG = (0,0,0)
  x = 12 # (WIDTH * 0.90)
  y = 530 # (HEIGHT * 0.80)
  windowSurface.fill(BACKG)
  windowSurface.blit(Icon, (x,y))
  pygame.display.set_icon(Icon)
  pygame.display.update()
  pygame.display.flip()
  
  # Music/Sound effects
  pygame.mixer.init()
  pygame.mixer.music.load('/home/pi/projects/ruspi/main/audio/alert_1.wav')
  pygame.mixer.music.play()
  while pygame.mixer.music.get_busy() == True:
    continue

def writeText(path, text, xoffset, x, y, size, w, h, textCol, rectCol):
  switcher = {
    0: "/home/pi/projects/ruspi/main/fonts/LCARS.ttf",
    1: "/home/pi/projects/ruspi/main/fonts/LCARSGTJ2.ttf",
    2: "/home/pi/projects/ruspi/main/fonts/LCARSGTJ3.ttf",
  }
  #windowSurface.fill(BACKG) 
  #windowSurface.blit(Image, (x,y))  
  Rectangle = pygame.Rect(x-5, y, w, h*1.1)
  pygame.draw.rect(windowSurface,(rectCol),Rectangle,0) 
  font = pygame.font.Font(switcher.get(path, "/home/pi/projects/ruspi/main/fonts/LCARSGTJ2.ttf"), size)
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

def main():
  pygame.init()
  # Load UI
  prepareUI(0) 
  Labels()
  
  #Seial 1
  #-------
  ser1 = serial.Serial()
  ser1.port = "/dev/ttyACM0"
  ser1.baudrate = 9600
  ser1.timeout = 1            #non-block read
  ser1.setDTR(False)
  ser1.bytesize = serial.EIGHTBITS #number of bits per bytes
  ser1.parity = serial.PARITY_NONE #set parity check: no parity
  ser1.stopbits = serial.STOPBITS_ONE #number of stop bits
  
  try: 
    ser1.open()
  except Exception, e:
    print "error open serial port: " + str(e)
    pi.stop()
    exit()
    
  if ser1.isOpen():

    try:
      ser1.flushInput()  #flush input buffer, discarding all its contents
      ser1.flushOutput() #flush output buffer, aborting current output 
                        #and discard all that is in buffer
      while True:                  
        while ser1.in_waiting:
          response = ser1.readline()
          #print(datetime)
          #print(ser1.port + ">> " + response)

          try:			  
            root = ET.fromstring(response)
            """
            <GPS>
              <LAT>String</LAT>    0
              <LONG>String</LONG>  1
              <DATE>String</DATE>  2
              <TIME>String</TIME>  3
            </GPS>    
            <IMU
              <ROLL>String</ROLL>    0
              <PITCH>String</PITCH>  1
              <YAW>String</YAW>  2
            </IMU>  
            """
            if root.tag == 'GPS':
              LAT = root[0].text
              LONG = root[1].text
              DATE = root[2].text
              TIME = root[3].text
              
              #(path, text, x, y, size, w, h, r, g, b)
              x = 200
              y = 185
              size = 30
              w = 120
              offset = 43
              rec = 0, 102, 102
              txtCol = 255,255,255
              writeText(1, root[2].text, 0, x, y, size, w, size, txtCol, rec) #Red
              y = y+offset
              writeText(1, root[3].text, 0, x, y, size, w, size, txtCol, rec)  #Orange
              y = y+offset
              writeText(1, root[0].text, 0, x, y, size, w, size, txtCol, rec) #Purple
              y = y+offset
              writeText(1, root[1].text, 0, x, y, size, w, size, txtCol, rec) #Blue
              for child in root:
                #print(child.tag, child.text)
                pass
              
            if root.tag == 'IMU':
              ROLL = root[0].text
              PITCH = root[1].text
              YAW = root[2].text
              #Draw Cycle count
              x = 200
              y = 355
              size = 55
              w = 150
              offset = 60
              rec = 0,0,0
              txtCol = 29, 136, 133
              writeText(1, root[0].text, 0, x, y, size, w, size, txtCol, rec) 
              y = y+offset
              txtCol = 36, 168, 164
              writeText(1, root[1].text, 0, x, y, size, w, size, txtCol, rec)  #Orange
              y = y+offset
              txtCol = 87, 218, 215
              writeText(1, root[2].text, 0, x, y, size, w, size, txtCol, rec)
              print("PITCH: ", PITCH, " ROLL: ", ROLL, " YAW: ", YAW)
              
              pitch_angle=float(PITCH)
              pitch_angle= 90 + pitch_angle
              if pitch_angle > 60 and pitch_angle < 120:
                pass
                pitch_pwm = pitch_angle*(2000/180) + 0   # pwm = ang(sweep/180) +/- offset
                pi.set_servo_pulsewidth(pitch_servo, pitch_pwm)  
              
              roll_angle=float(ROLL)
              if roll_angle <= 0:
                roll_angle= 180 + roll_angle
                if roll_angle >= 165: 
                  roll_pwm = roll_angle*(2000/180) - 500   # pwm = ang(sweep/180) +/- offset
                  pi.set_servo_pulsewidth(roll_servo, roll_pwm)  
                print("ROLL Calc.: ", roll_angle)
              elif roll_angle > 0:
                roll_angle= 180 + roll_angle
                if roll_angle <= 195: 
                  pwm = roll_angle*(2000/180) - 500   # pwm = ang(sweep/180) +/- offset
                  pi.set_servo_pulsewidth(roll_servo, pwm)
                print("ROLL Calc.: ", roll_angle)            
          
          except Exception, e:
            print "[ERR] unable to read from serial port: " + str(e)
            pass
            
        for event in pygame.event.get():
          if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
              pi.stop()			  
              ser1.close()
              print(">> Serial Port " + ser1.port + " closed")  
              pygame.mixer.music.load('/home/pi/projects/ruspi/main/audio/deactivation_complete.wav')
              pygame.mixer.music.play()
              while pygame.mixer.music.get_busy() == True:
                continue
              pygame.quit()
              sys.exit()
              quit()
                      

    except Exception, e1:
      print "[ERR] Error communicating...: " + str(e1)
      pass
 
  else:
    print "[ERR] Cannot open serial ports"
    pi.stop()
    pygame.quit()
    sys.exit()
    pass
    
# Execute MAIN
if __name__ == '__main__':
  main()
  
