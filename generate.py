"""
Script Name:
    generate.py
    
Author:
    Simon Gunkel
    
Date:
    May 25, 2023

Version:
    2.0.

Description:
    DelAtrUco - Generator

    This tool is to generate ArUco markers to measure delay.

    Update v2:
    - detach marker generation into own thread (if we touch a window the output will freeze)
    - detach virtual webcam output into own thread (if we touch a window the output will freeze)
    
Licensing Terms: 
    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.    
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import pyvirtualcam
import win_precise_time as wpt
import queue
import threading
import argparse


# create a arg parser
parser = argparse.ArgumentParser(description='DelAyrUco - Analyser')

#get all the arguments we need!
parser.add_argument('--framerate', '-f', type=int, default=30, help='framerate in frames / sec')
parser.add_argument('--arucolayout', '-l', type=int, default=1, help='output resolution and layout 0. (VGA) / 1. 720p / 2. 512x1024')

args = parser.parse_args()
# get values from argparse

#get the arguments
# --framerate   - r framerate
# --arucolayout - a 0. (VGA) / 1. 720p / 2. 512x1024
framerate = args.framerate
arucolayout = args.arucolayout

#init
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

marker_l_dict = {} # values from 0 - 1000 image for the top
marker_h_dict = {} # values from 1000 - 1000000 image for the bottom or right


#note: 512 is arucolayout 2
marker_img_size = 512
marker_img_shape = (512, 512, 3)
output_img_shape = (1024, 512, 3)
border_size = 64
if (arucolayout == 0):
    marker_img_size = 640 // 2
    border_size = marker_img_size // 8
    marker_img_shape = (480, marker_img_size, 3)
    output_img_shape = (480, 640, 3)
elif (arucolayout == 1):
    marker_img_size = 1280 // 2
    border_size = marker_img_size // 8
    marker_img_shape = (720, marker_img_size, 3)
    output_img_shape = (720, 1280, 3)

#pre-generate all images for the markers
for i in range(1000):
    #if we are on VGA or 720p resolution we like to center the markers on the y-axis
    height_offset = (marker_img_shape[0] - marker_img_size) // 2
    
    marker = aruco.generateImageMarker(dictionary, i, border_size*6, border_size)
    l = np.zeros(marker_img_shape, dtype=np.uint8)
    l.fill(255)
    l[border_size + height_offset:border_size * 7 + height_offset, border_size:border_size * 7][marker == 0] = (255, 0, 0)
    marker_l_dict[i] = l
    r = np.zeros(marker_img_shape, dtype=np.uint8)
    r.fill(255)
    r[border_size + height_offset:border_size * 7 + height_offset, border_size:border_size * 7][marker == 0] = (0, 0, 255)
    marker_h_dict[i] = r

# define marker parameters
marker_l_id = -1
marker_h_id = 0

print(f'initiate virtual camera - width={output_img_shape[1]} and height={output_img_shape[0]}')
virtual_cam = pyvirtualcam.Camera(width=output_img_shape[1], height=output_img_shape[0], fps=framerate, fmt=pyvirtualcam.PixelFormat.BGR, print_fps=True, device="OBS Virtual Camera")

last_frame_timestamp = wpt.time()
start_time_deltas = [] #array to create a good scheduling offset

# Global queue variables to synchronize frames across threads
vcam_frames = queue.Queue()
render_frames = queue.Queue()

is_running = True

def marker():
    global is_running, marker_l_id, marker_h_id, arucolayout, marker_l_dict, marker_h_dict, last_frame_timestamp, vcam_frames, render_frames
    while is_running:
        marker_l_id += 1

        if marker_l_id >= 1000:
            marker_l_id = 0
            marker_h_id += 1

        if marker_h_id >= 1000:
            marker_h_id = 0

        img = None
        if arucolayout == 2:
            #if we are on 512x1024 image size we want to do a up / down layout and combine the image via bytes for high efficiency
            img = np.frombuffer(marker_l_dict[marker_l_id].tobytes() + marker_h_dict[marker_h_id].tobytes(), dtype=np.uint8)
            img.shape = (1024, 512, 3)
        else:
            img = np.hstack([marker_l_dict[marker_l_id],marker_h_dict[marker_h_id]])

        timestamp = wpt.time()
        wpt.sleep(1/framerate - (timestamp - last_frame_timestamp))
        last_frame_timestamp = wpt.time() - 0.0004 #NOTE WE ADD 0.4ms to compensate for the wpt.time() command itself
        vcam_frames.put_nowait(img)
        render_frames.put_nowait(img)
        
def vcam():
    global is_running, vcam_frames, virtual_cam
    while is_running:
        # Get the next image
        img = vcam_frames.get()

        while not vcam_frames.empty():
            #it is expected that we always act on every frame, we can consider this as an error (can the virtual webcam block!?)
            #TODO: check if we like to change this loop with a assert
            img = vcam_frames.get()

        #push the frame to the virtual webcam
        virtual_cam.send( img )

#lets start the threads
marker_thread = threading.Thread(target=marker)
marker_thread.start()

# Start your method in a new thread
vcam_thread = threading.Thread(target=vcam)
vcam_thread.start()

while is_running:
    # Get the next image
    img = render_frames.get()

    while not render_frames.empty():
        #this thread loop might block if we touch (e.g., move) an window
        img = render_frames.get()

    #render the frame on screen
    cv2.imshow('DelAyrUco - Generator', img )
    
    key = cv2.waitKey(1)
    
    # close window if 'q' or 'esc' is pressed
    if key == ord('q') or key == 27:  # 27 is the ASCII code for 'esc'
        cv2.destroyAllWindows()
        is_running = 0

marker_thread.join()
vcam_thread.join()
exit()
    


    
