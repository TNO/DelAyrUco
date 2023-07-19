"""
Script Name:
    analyse.py
    
Author:
    Simon Gunkel
    
Date:
    May 25, 2023

Version:
    1.1.

Description:
    DelAyrUco - Analyser

    This tool is to analyze recording files that are created with the DelAyrUco generator.

    Important considerations

    - We always expect to have the reference marker (sometimes referred in the code as calibration marker) in the image, this marker should capture the direct feed from the webcam picture generator
    - We only change condition if the calibration ONLY is visible for a period of time (at least one frame)
    - We only use 1000 x 1000 maximum IDs - this means if you run the program for longer then 1000000 / fps sec (for 30fps = 555 min) accuracy cannot be guaranteed

    How does this work:
    1. Open the video file
    2. Initiate with an image only showing the reference marker (this will estimate the layout / if nor set in starting parameters)
    3. Once more markers are visible initiate the condition
    4. measure delays for each DelAyrUco MArker combo (delta between reference makers)
    5. Do so until end of file or we have only reference marker in a frame (in the later, increase condition and go back to step 2)
    6. Finally print results.

    Results are calculated as Mean and standard deviation, with each condition and tile.

Updates: 
    1.1 fixes for vertical layout

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
import time
import argparse
import os


# create a arg parser
parser = argparse.ArgumentParser(description='DelAtrUco - Analyser')

#get all the arguments we need!
parser.add_argument('--inputfile', '-i', type=str, help='filename of file to analyze')
parser.add_argument('--outputmode', '-o', type=int, default=0, help='format the output as 0. CSV / 1. TEX (default = 0)')
parser.add_argument('--condition_size', '-c', type=int, default=0, help='if set, fixed condition size')
parser.add_argument('--marker_layout', '-l', type=int, default=-1, help='if set, fixes the layout of the DelAtrUco markers: 0 - horizontal marker layout; 1 - vertical (top/bottom) marker layout')
parser.add_argument('--videooffset', type=int, default=0, help='ignore the first part of the video (in milliseconds)')
parser.add_argument('--debugoutput', '-d', type=int, default=0, help='do debug prints')


args = parser.parse_args()
# get setting values from argparse
inputfilename = args.inputfile
outputmode = args.outputmode
fixed_condition_size = args.condition_size # default should be 0
layout = args.marker_layout # 0 - horizontal marker layout; 1 - vertical (top/bottom) marker layout
debug = args.debugoutput

print(f'Starting to analyse file {inputfilename} with mode {outputmode} and fixed condition {fixed_condition_size}')

# DEBUG settings
videooffset = args.videooffset # start video after a period of time, currently not used
max_id_distance = 300 # filter out any wrong values that do not fall into a 10sec delay time / filtering is needed to filter out wrong detections because of distortions like motion blur
min_marker_size = 1000 # filter out marker that are smaller (in square pixel)

# open video file
cap = cv2.VideoCapture(inputfilename)
exit_programm = 0 # boolean to indicate program operation

#frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
#frame_count = int(cap.get(cv.CAP_PROP_POS_FRAMES))

# init the aruco detection
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
parameters =  cv2.aruco.DetectorParameters()
parameters.errorCorrectionRate = 2
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# global values for reference marker
is_calibration = True
do_calibration_snapshot = False
calibration_l_marker_corners = None #marker for lower values 0-1000
calibration_h_marker_corners = None #marker for higher values 1000-1000000

base_timestamp = None
base_index = 0

last_calibration_marker_id = -1
calibration_marker_id = -1
last_calibration_timestamp = 0
calibration_lostframes = 0
calibration_timestamp = {}

# global values for the condition(s)
is_condition_initiated = False # we need some actions only once at the beginning to setup a condition
current_condition = 0 # index of the condition we are currently in (0 - init)
current_condition_tiles = 0 # number of tiles in the current condition
latency_conditions = {} # all latency numbers of all conditions and tiles
tile_l_marker_corners = {} # to properly track marker across one condition we need to store the location of each marker
tile_h_marker_corners = {} # to properly track marker across one condition we need to store the location of each marker

while not exit_programm:
    # read frame from video file
    ret, frame = cap.read()

    if not ret:
        print("video has ended - exit")
        exit_programm = 1
        continue

    # get current position of video file (in milliseconds)
    video_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

    if video_timestamp < videooffset:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(gray)

    if debug: #DEBUG
        print(f'markerCorners:{markerCorners}')
        print(f'markerIds:{markerIds}')
        print(f'rejectedCandidates:{rejectedCandidates}')

    # draw the detected markers on the video image frame
    out_image = cv2.aruco.drawDetectedMarkers(frame.copy(), markerCorners, markerIds)

    #filter out marker that are too small (i.e. possible misdirections / e.g. window bars etc.)
    indexes_to_remove = []
    for index, corners in enumerate(markerCorners):
        # Calculate the surface area for a marker with the Shoelace formula
        n = len(corners[0])
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += corners[0][i][0] * corners[0][j][1]
            area -= corners[0][j][0] * corners[0][i][1]
        area = abs(area) / 2 #calculate marker area in square pixels

        if debug: # DEBUG
            position = (int(corners[0][2][0]), int(corners[0][2][1]))  # (x, y) coordinates of the text position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            color = (0, 255, 0)  # Text color in BGR format (white in this case)
            thickness = 1  # Line thickness
            cv2.putText(out_image, str(area), position, font, font_scale, color, thickness)

        if area < min_marker_size:
            indexes_to_remove.append(index)
    
    # Remove all information for markers we like to filter out
    markerIds_array = []
    for index, markerId in enumerate(markerIds):
        if not index in indexes_to_remove:
            markerIds_array.append(markerId)
    # markerCorners is a tuple so we like to recreate it as an array without the filtered marker indexes
    markerCorners_array = []
    for index, corners in enumerate(markerCorners):
        if not index in indexes_to_remove:
            markerCorners_array.append(corners)
    markerIds = markerIds_array
    markerCorners = markerCorners_array

    if debug: #DEBUG
        for index, corners in enumerate(markerCorners):
            position = (int(corners[0][0][0]), int(corners[0][0][1]))  # (x, y) coordinates of the text position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            color = (0, 255, 0)  # Text color in BGR format (white in this case)
            thickness = 2  # Line thickness
            cv2.putText(out_image, str(index // 2), position, font, font_scale, color, thickness)

    #check if we have markers
    if markerIds is None:
        continue # we dont act on images without markers
    num_markers = len(markerIds)
    if (num_markers <=0):
        continue 
        # NOTE: we dont act on images without markers
        # NOTE: we always expect calibration markers in the image
    
    #initiate calibration marker
    # NOTE: we always assume the calibration marker is at the same place
    # NOTE: we assume the first image frame in the video only shows the calibration marker
    if calibration_h_marker_corners is None:
        #find top - left marker
        #note in calibration there should be only one marker on screen
        top_corner_y = -1
        top_corner_x = -1
        top_l_corner_index = -1
        calibration_l_marker_index = -1
        calibration_h_marker_index = -1

        for index, corners in enumerate(markerCorners):
            #print (f'{index} - {corners}')
            if top_corner_y <= -1 or top_corner_y > corners[0][0][1] or top_corner_x > corners[0][0][0]:
                top_corner_y = corners[0][0][1]
                top_corner_x = corners[0][0][0]
                top_l_corner_index = index
        
        # set the calibration marker indexes
        if (top_l_corner_index == 0):
            calibration_l_marker_index = 0
            calibration_h_marker_index = 1
        else:
            calibration_l_marker_index = 1
            calibration_h_marker_index = 0

        #layout it set to auto-detect, lets detect
        if layout < 0:
            #lets check if L marker is top of H marker
            if markerCorners[calibration_l_marker_index][0][0][1] < markerCorners[calibration_h_marker_index][0][0][1]:
                layout = 1
            else:
                layout = 0

            
        elif layout == 0:
            calibration_l_marker_index = 1
            calibration_h_marker_index = 0
        else: # layout == 1:
            print(f'Lets fix layout 1')
            calibration_l_marker_index = 1
            calibration_h_marker_index = 0
            
        print(f'Calibration marker found - layout is set to {layout}')

        calibration_l_marker_corners = markerCorners[calibration_l_marker_index]
        calibration_h_marker_corners = markerCorners[calibration_h_marker_index]
        
        print(f'calibration_l_marker_corners {calibration_l_marker_corners}')
        print(f'calibration_h_marker_corners {calibration_h_marker_corners}')
    
    #find the calibration marker index
    calibration_l_marker_index = -1
    calibration_h_marker_index = -1
    if (num_markers == 2 and is_condition_initiated):
        # is we have 2 markers we move out of the current condition
        print(f'... Condition ended')
        is_condition_initiated = False
    #find calibration marker based on previous corner detection
    for index, corners in enumerate(markerCorners):
        if (corners == calibration_l_marker_corners).all():
            calibration_l_marker_index = index
        elif (corners == calibration_h_marker_corners).all():
            calibration_h_marker_index = index

    if ( calibration_l_marker_index == -1 or calibration_h_marker_index == -1):
        # NOTE: we always asume the calibration marker in the image and at the same place
        print(f'video has no calibration marker - exit')
        exit_programm = 1
        continue

    for index, corners in enumerate(markerCorners):
        if (index in [calibration_l_marker_index]):
            position = (int(corners[0][0][0]), int(corners[0][0][1]))  # (x, y) coordinates of the text position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            color = (0, 255, 0)  # Text color in BGR format (white in this case)
            thickness = 2  # Line thickness
            cv2.putText(out_image, 'calib-l', position, font, font_scale, color, thickness)
        if (index in [calibration_h_marker_index]):
            position = (int(corners[0][0][0]), int(corners[0][0][1]))  # (x, y) coordinates of the text position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            color = (0, 255, 0)  # Text color in BGR format (white in this case)
            thickness = 2  # Line thickness
            cv2.putText(out_image, 'calib-h', position, font, font_scale, color, thickness)

    calibration_marker_id = markerIds[calibration_l_marker_index][0] + (markerIds[calibration_h_marker_index][0] * 1000)
    #compensate if h marker runs over 1000
    if calibration_marker_id < 0: #we run over the 1.000.000 line
        calibration_marker_id += 1000000

    #set the base timestamp and index
    #NOTE: we are on the very 1st frame here
    if base_timestamp is None:
        base_timestamp = video_timestamp
        last_calibration_timestamp = video_timestamp
        base_index = calibration_marker_id
        last_calibration_marker_id = calibration_marker_id
        # print(f'set base_timestamp - {base_timestamp} - index {base_index}')
        #cv2.waitKey(1000)
        continue #do not use first frame

    calibration_delta = video_timestamp - last_calibration_timestamp
    if (calibration_marker_id > (last_calibration_marker_id)):
        calibration_timestamp[calibration_marker_id] = video_timestamp
        last_calibration_marker_id = calibration_marker_id
    
    # lets deal with latency measures
    # 1st check if we have enough markers
    if (num_markers > 2):
        
        if not is_condition_initiated:
            #intitate for a new condition
            if (num_markers % 2 > 0):
                continue # we dont have all makers
            
            if (fixed_condition_size > 0):
                if (num_markers != (fixed_condition_size * 2)):
                    print (f'Not the right number of markers to init {num_markers} vs. {fixed_condition_size* 2}')
                    continue # we need to initialize when we have correct tiles

            # 2nd identify how many tiles = marker / 2 - 1 (calibration markers do not count)
            num_tiles = num_markers // 2
            
            # We have a new condition
            current_condition += 1
            current_condition_tiles = num_tiles

            latency_conditions[current_condition] = {}

            # find the position of all markers
            # 1. split markers to lower (blue) and higher (red)
            # 2. match tiles based on center distance and layout (blue directly on top, or on left)
            l_marker_indexes = []
            h_marker_indexes = []
            red = np.array((0,0,255)) # NOTE: we are in bgr color space
            blue = np.array((255,0,0))
            for index, corners in enumerate(markerCorners):
                # print(corners)
                top_corner_y = int(corners[0][0][1]) + 2 # lets add one pixel distance to be sure we catch a color pixel
                top_corner_x = int(corners[0][0][0]) + 2 # lets add one pixel distance to be sure we catch a color pixel
                top_corner_color = np.array(frame[top_corner_y, top_corner_x])
                # print (f'index {index} with color {top_corner_color}')
                if (abs(top_corner_color - blue).sum() < 255): # so far 255 gives good results, even with discolorations in webcam capture
                    l_marker_indexes.append(index)
                elif (abs(top_corner_color - red).sum() < 255): # so far 255 gives good results, even with discolorations in webcam capture
                    h_marker_indexes.append(index)
                else:
                    pass # TODO: CHECK what to do here / assert!?

            tile_index = -1
            for h_index in h_marker_indexes:
                tile_index += 1
                h_center = np.mean( np.array(markerCorners[h_index]) , axis=(0, 1)).astype(int)
                h_size = np.max(np.array(markerCorners[h_index]), axis=(0, 1)).astype(int) - np.min(np.array(markerCorners[h_index]), axis=(0, 1)).astype(int)
                for l_index in l_marker_indexes:
                    l_center = np.mean( np.array(markerCorners[l_index]) , axis=(0, 1)).astype(int)
                    
                    if layout <= 0: # VGA / 720 horizontal layout
                        # print (f'h{h_index}-{h_center} l{l_index}-{l_center}> abs(({h_center[1]} - {l_center[1]})) < 10 and ({h_center[0]} > {l_center[0]}) and ({h_center[0]} - {l_center[0]}) < ({h_size[0]} * 2)')
                        if abs((h_center[1] - l_center[1])) < 10 and (h_center[0] > l_center[0]) and (h_center[0] - l_center[0]) < (h_size[0] * 2):
                            tile_l_marker_corners[tile_index] = markerCorners[l_index]
                            tile_h_marker_corners[tile_index] = markerCorners[h_index]
                    else: # 512 vertical layout
                        if abs((h_center[0] - l_center[0])) < 10 and (h_center[1] > l_center[1]) and (h_center[1] - l_center[1]) < (h_size[1] * 2):
                            tile_l_marker_corners[tile_index] = markerCorners[l_index]
                            tile_h_marker_corners[tile_index] = markerCorners[h_index]
        
            # print (f'{current_condition_tiles}: tile_l_marker_corners {tile_l_marker_corners} and tile_h_marker_corners {tile_h_marker_corners}')

            do_calibration_snapshot = True
            is_condition_initiated = True
            print(f'Condition {current_condition} initialized ... tiles = {num_tiles}')
            cv2.waitKey(1000)

        tile_l_marker_corner_index = {}
        tile_h_marker_corner_index = {}
        for index, corners in enumerate(markerCorners):
            # find the tile
            for tile_index in tile_l_marker_corners:
                tile_corners = tile_l_marker_corners[tile_index]
                if (np.absolute(abs((corners - tile_corners)).sum()) < 20): 
                    tile_l_marker_corner_index[tile_index] = index
                    #print (f'corners {corners} match tile_corners {tile_corners}')
                #else:
                #    print (f'corners {corners} not tile_corners {tile_corners}')
            for tile_index in tile_h_marker_corners:
                tile_corners = tile_h_marker_corners[tile_index]
                if (np.absolute(abs((corners - tile_corners)).sum()) < 20): 
                    tile_h_marker_corner_index[tile_index] = index
                    #print (f'corners {corners} match tile_corners {tile_corners}')

        for tile_index in range(current_condition_tiles):
            if not (tile_index in tile_l_marker_corner_index and tile_index in tile_h_marker_corner_index):
                print (f'tile_index {tile_index} not in  {tile_l_marker_corner_index} or {tile_h_marker_corner_index}')
                continue #we did not have both markers for this tile!

            tile_h_marker_index = tile_h_marker_corner_index[tile_index]
            tile_l_marker_index = tile_l_marker_corner_index[tile_index]

            for index, corners in enumerate(markerCorners):
                if (index in [tile_h_marker_index]):
                    position = (int(corners[0][3][0]), int(corners[0][3][1]))  # (x, y) coordinates of the text position
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    color = (0, 255, 0)  # Text color in BGR format (white in this case)
                    thickness = 2  # Line thickness
                    cv2.putText(out_image, f'Tile {tile_index} - H', position, font, font_scale, color, thickness)

                if (index in [tile_l_marker_index]):
                    position = (int(corners[0][3][0]), int(corners[0][3][1]))  # (x, y) coordinates of the text position
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    color = (0, 255, 0)  # Text color in BGR format (white in this case)
                    thickness = 2  # Line thickness
                    cv2.putText(out_image, f'Tile {tile_index} - L', position, font, font_scale, color, thickness)

            # tile_index = tile_index #str(tile_index) # lets use a string for dict access
            #if any(value in [tile_h_marker_index, tile_l_marker_index] for value in [calibration_l_marker_index, calibration_h_marker_index]):
            #    continue #skip calibration

            tile_marker_id = markerIds[tile_l_marker_index][0] + (markerIds[tile_h_marker_index][0] * 1000)

            # print (f'tile_index {tile_index} - {tile_marker_id}')
            #print (f'tile{tile_index}: {tile_marker_id} - L={tile_l_marker_index} + H={tile_h_marker_index}')
            #compensate if h marker runs over 1000
            if tile_marker_id < 0: #we run over the 1.000.000 line
                tile_marker_id += 1000000

            if not (tile_index in latency_conditions[current_condition]):
                latency_conditions[current_condition][tile_index] = {}
                latency_conditions[current_condition][tile_index]["last_id"] = tile_marker_id # this means we ignore the first frame / NOTE we could also set -1 here to include it
                latency_conditions[current_condition][tile_index]["lost_frames"] = 0
                latency_conditions[current_condition][tile_index]["values"] = []

            if tile_marker_id != latency_conditions[current_condition][tile_index]["last_id"]:
                if tile_marker_id < latency_conditions[current_condition][tile_index]["last_id"]:
                    print (f'skip frame ... tile_marker_id < latency_conditions')
                    pass #latency_conditions[current_condition][tile_index]["lost_frames"] += 1 
                    # for now pass but we might want to increase lost frame counter
                else:
                    if abs(tile_marker_id - latency_conditions[current_condition][tile_index]["last_id"]) >= max_id_distance:
                        print (f'skip frame ... filter for most likely wrong ArUco matches')
                        pass #ignore this case for now / this is the filter for most likely wrong ArUco matches (based on distortions like motion blur)
                    elif tile_marker_id in calibration_timestamp:
                        if tile_marker_id > latency_conditions[current_condition][tile_index]["last_id"] + 1:
                            latency_conditions[current_condition][tile_index]["lost_frames"] += (tile_marker_id - latency_conditions[current_condition][tile_index]["last_id"]) - 1
                        latency = video_timestamp - calibration_timestamp[tile_marker_id]
                        latency_conditions[current_condition][tile_index]["values"].append(latency)
                        # print (f'Condition {current_condition} - Tile {tile_index} - latency {latency}')
                        if latency > 10000:
                            print (f'Condition {current_condition} - Tile {tile_index} - latency {latency}')
                            print (f'tile_marker_id {tile_marker_id} - last_id {latency_conditions[current_condition][tile_index]["last_id"]}')
                            cv2.waitKey(0)
                    else:
                        print (f'Marker not found in calibration {tile_marker_id}')
                        if tile_marker_id > latency_conditions[current_condition][tile_index]["last_id"] + 1:
                            latency_conditions[current_condition][tile_index]["lost_frames"] += (tile_marker_id - latency_conditions[current_condition][tile_index]["last_id"]) - 1
                        else:
                            latency_conditions[current_condition][tile_index]["lost_frames"] += 1

                    latency_conditions[current_condition][tile_index]["last_id"] = tile_marker_id
        
        #print (f'latency_conditions {latency_conditions}')
            

    # display the screenshot with detected markers
    cv2.imshow("DelAyrUco Analyser", out_image ) #cv2.resize(out_image, (640, 480)))
    key = cv2.waitKey(1)
    if (do_calibration_snapshot):
        do_calibration_snapshot = False
        filename_wihtout_ending = os.path.splitext(inputfilename)[0]
        cv2.imwrite(f'{filename_wihtout_ending}_{current_condition}.png', out_image)

    if key == ord('q') or key == 27:  # 27 is the ASCII code for 'esc'
        print(f'user requested exit wiht key {key}')
        exit_programm = 1
        continue
    elif key == ord('p'):
        cv2.waitKey(0)

csv_output = 'file, condition, tile, latency, latency std, num of samples, lost frames \n'
latex_output = 'file & condition & tile & latency & latency std & num of samples & lost frames \% \\\\ \n'

print ('Analysis done ...')
print ('')
print ('=== RESULTS ===')

for condition in latency_conditions:
    for tile_index in latency_conditions[condition]:
        condition_values_np = np.array(latency_conditions[condition][tile_index]["values"])
        #print(f'Condition {condition} {condition_values_np}')
        if condition_values_np.size > 0:
            print(f'Condition {condition} - Tile {tile_index} - [{len(condition_values_np)}] - frame_delta:{np.mean(condition_values_np):.2f} - min {np.min(condition_values_np):.2f} - max {np.max(condition_values_np):.2f} - std:{np.std(condition_values_np):.2f} - lost frames: {latency_conditions[condition][tile_index]["lost_frames"]}')
            latex_output += f'{inputfilename} & {condition} & {tile_index} & {np.mean(condition_values_np):.2f} & {np.std(condition_values_np):.2f} & {len(condition_values_np)} & {latency_conditions[condition][tile_index]["lost_frames"]} \\\\ \n'
            csv_output += f'{inputfilename}, {condition}, {tile_index}, {np.mean(condition_values_np):.2f}, {np.std(condition_values_np):.2f}, {len(condition_values_np)}, {latency_conditions[condition][tile_index]["lost_frames"]} \n'


print ('')
print ('=== LATEX ===')
print (latex_output)
print ('=== ----- ===')

print ('')
print ('===  CSV  ===')
print (csv_output)
print ('=== ----- ===')

#cleanup befor exit
cv2.destroyAllWindows()
# release the video file
cap.release()
