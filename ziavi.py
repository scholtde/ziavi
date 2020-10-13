#!/usr/bin/env python
import serial
import time
import random

RX_BUFFER_SIZE = 128

verbose = True
l_count = 0
g_count = 0
c_line = []

# Open grbl serial port
s = serial.Serial('/dev/ttyACM0',115200)

# Wake up grbl
s.write(b"\r\n\r\n")
time.sleep(2)   # Wait for grbl to initialize 
s.flushInput()  # Flush startup text in serial input

# CNC setttings
l_block = bytes("$120=2", 'utf-8')
s.write(l_block + b'\n') # Send g-code block to grbl
time.sleep(1)
l_block = bytes("$121=2", 'utf-8')
s.write(l_block + b'\n') # Send g-code block to grbl
time.sleep(1)

# Stream g-code to grbl
while True:
    l_count += 1 # Iterate line counter
    x = str(random.randint(0,10))
    y = str(random.randint(0,10))
    speed = str(random.randint(200,250))
    l_block = bytes("$J=X"+x+" Y"+y+" F"+speed, 'utf-8')
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
    
    #print(b'Sending: ' + l_block)
    #s.write(l_block + b'\n') # Send g-code block to grbl
    #grbl_out = s.readline().strip() # Wait for grbl response with carriage return
    #print(b' : ' + grbl_out.strip())

# Wait for user input after streaming is completed
print("G-code streaming finished!\n")
print("WARNING: Wait until grbl completes buffered g-code blocks before exiting.")
input("  Press <Enter> to exit and disable grbl.") 

# Close file and serial port
f.close()
s.close()    
