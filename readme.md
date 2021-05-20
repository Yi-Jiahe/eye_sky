# eye-sky

eye-sky is a Python port of the multiple-object multiple-camera tracker, SPOTIT3D developed by Dr Sutthiphong, Srigrarom. Alongside the original functionality implemented in MATLAB, eye-sky also features improvements such as filtering to improve object detection, intuitive visualization, spatial positioning and 3D triangulation.

## Publications
Two of the ideas worked on in eye-sky have lead to the publications:

J.H. Yi, K.H. Chew and S. Srigrarom, “Enabling Continuous Drone Tracking Across Translational Scene Transitions through Frame-Stitching”, 2021 Second International Symposium on Instrumentation, Control, Artificial Intelligence, and Robotics (ICA-SYMP), 20-22 January 2021, Bangkok, Thailand.

J.H. Yi and S. Srigrarom, "Near-parallel binocular-like camera pair for multi-drone detection and 3D localization", IEEE 16th International Conference on Control, Automation, Robotics and Vision, (ICARCV 2020), 13-15 December 2020, Shenzhen, P.R.China.

## Overview
The project began with the following scripts
 - object_tracker, which handled frame I/O, image transformations, background_subtraction, object detection and tracking
 - camera_stabilizer scripts which handles camera stabilization by optical flow methods
 - track_visualization was added in to handle post-processing and plotting of tracks
 - automatic_brightness which adjusted the gamma and alpha values in preproccesing

The real_time_object_detection package was developed to process live video feed and provide a user interface for running the scripts.

The Binocular Camera package was developed for 3D triangulation with a binocular set-up.

The multiple indoor views and real-time multiple camera folders contain some tests performed with footage captured indoors and a simulated multiple drone setup.

The different_object_trackers package tested different object trackers offered in OpenCV for comparison.

Finally the Tello scripts folder contains an attempt to run the object detection and tracking from a Tello drone using the video feed.

## Dependencies
opencv-python

numpy(in opencv-python)

FilterPy

scipy(in FilterPy)

matplotlib (in FilterPy)	