#!/usr/bin/env python
import serial
import time

# Open grbl serial port
s = serial.Serial('/dev/ttyACM0',115200)

# Wake up grbl
s.write(b"\r\n\r\n")
time.sleep(2)   # Wait for grbl to initialize 
s.flushInput()  # Flush startup text in serial input

j_x = 100
j_y = 100
speed = str(200)
l_block = bytes("$J=X"+str(j_x)+" Y"+str(j_y)+" F"+speed, 'utf-8')
print("SND: " + str(l_block))
s.write(l_block + b'\n') # Send g-code block to grbl

l = bytes('!', 'utf-8')
print(l)
#print(b'Sending: ' + l)
s.write(l) # Send g-code block to grbl
grbl_out = s.readline() # Wait for grbl response with carriage return
print(b' : ' + grbl_out.strip())

j_x += 100
j_y += 100
speed = str(200)
l_block = bytes("$J=X"+str(j_x)+" Y"+str(j_y)+" F"+speed, 'utf-8')
print("SND: " + str(l_block))
s.write(l_block + b'\n') # Send g-code block to grbl

time.sleep(1)
l = bytes('!', 'utf-8')
print(l)
#print(b'Sending: ' + l)
s.write(l) # Send g-code block to grbl
grbl_out = s.readline() # Wait for grbl response with carriage return
print(b' : ' + grbl_out.strip())

# Wait here until grbl is finished to close serial port and file.
input("  Press <Enter> to exit and disable grbl.") 

# Close file and serial port
f.close()
s.close()    
