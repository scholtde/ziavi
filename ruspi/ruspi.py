import os
import time
import keyboard
import pygame, sys
import pygame.locals
                                 
import time  
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
    WIDTH = 500
    HEIGHT = 500
    windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
     
  Icon = pygame.image.load('/home/nvidia/projects/ruspi/img/RCT_Logo_205x71l.png')
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
  pygame.mixer.music.load('/home/nvidia/projects/ruspi/audio/alert_1.wav')
  pygame.mixer.music.play()
  while pygame.mixer.music.get_busy() == True:
    continue

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

def main():
  pygame.init()
  # Load UI
  prepareUI(1) 
  Labels()
  while True:
    for event in pygame.event.get():
      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
          #pi.stop()			  
          #ser1.close()
          #print(">> Serial Port " + ser1.port + " closed")  
          pygame.mixer.music.load('/home/nvidia/projects/ruspi/audio/deactivation_complete.wav')
          pygame.mixer.music.play()
          while pygame.mixer.music.get_busy() == True:
            continue
          pygame.quit()
          sys.exit()
          quit()   
# Execute MAIN
if __name__ == '__main__':
  main()
  
