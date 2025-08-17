
#  Vehicle Counter using YOLO & OpenCV
This project is a real-time vehicle detection and counting system built with YOLOv11, OpenCV, and Python.
It allows you to mark reference lines on a video feed, count vehicles as they cross those lines, and maintain detailed logs for analysis.
The whole project is also Dockerized for easy setup and deployment.

## ✨ Features
-Real-time Vehicle Detection

-Uses YOLOv11 to detect vehicles (car, truck, bus, motorbike, bicycle)

-Line-based Vehicle Counting

-Draw one or more reference lines — whenever a vehicle crosses, it increments the count

-Interactive Line Marking

-Pauses the video during line marking

-Draw as many lines as needed before resuming

-Logging System

-Counts and events are logged with timestamps

-Logs are saved in /logs as .log files


## Modular Code Structure

main.py → Entry point (runs detection + counting)

utils.py → Utility functions (drawing, logging, etc)

logs/ → Stores logs for each run
