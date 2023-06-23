# DelAyrUco

## Description
DelAyrUco (wordplay on delay and ArUco), is a new delay measurement tool/method, including simple python based program code to generate markers and measure delays in multimedia applications (e.g. real-time communication). In some way its similar to the also open source tool VideoLat (https://www.cwi.nl/en/results/software/videolat-measuring-end-to-end-delays-in-a-v-systems/), but with some key differences:
-	analysis done post measurement 
-	Frame accurate measure, measuring lost frames and simplify measurement duration 
-	significantly decreased in std (error margin) compared to state of the art 
-	measuring multiple streams at the same time 
-	simply python code allowing easy extension and batch process testing 

An explanation of DelAyrUco can be found in the following paper: [add link after paper is published]

We expect that when you using DelAyrUco (especially for scientific work) that you give apropriate credits and cite the paper:
[add citation after paper is published]

### DelAtrUco - Generator - generate.py
The program to create a constant stream of ID ArUco marker for the delay estimation.

### DelAyrUco - Analyser - analyse.py
The program to analyze recording files that include the markers generated with the DelAyrUco Generator.

Important considerations: 
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

## Installation / Requirements
A python environment can be created using pip, using [requirements.txt](requirements.txt)

pip install -r requirements.txt

We developed and tested DelAyrUco with Python 3.10.9

### Python packages
DelAyrUco uses the following packes
- numpy: https://numpy.org/
- opencv: https://docs.opencv.org/4.x/
- pyvirtualcam: https://github.com/letmaik/pyvirtualcam
- win_precise_time: https://github.com/zariiii9003/win-precise-time

## Usage

A video instruction on how to use DelAyrUco can be found here: https://youtu.be/tqHe1PDgjxM

### Generator 
Simply run with:
- python generate.py

### Analyser
To run the example in the repo:
- python analyse.py -i test_example_jitsi_c4_30fps_1.mkv -c 4

## Support
For any support and improvement suggestions please use the issue function in github.

## Roadmap
some posibilties for improvement:
- add motion deblur
- test DelAyrUco in controled environment (e.g., video conferncing + fixed network delay)
- support for linux and MacOS

## Authors and acknowledgment
Thanks to xxx, xxx, xxx for helping to open source this tool.

## License
Apache License Version 2.0

Copyright (c) 2023 TNO

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

## Project status
Project is currently under submission for publication in ACM MM.

Anyone is welcome to use, build on top and improve DelAyrUco.
