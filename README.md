# Smart Surveillance System

An AI-powered real-time surveillance system built using **YOLOv8**, **Flask**, and a **web dashboard**.

## Features
- Real-time object detection using YOLOv8
- Live camera feed in browser
- Detection history tracking
- Alert system for suspicious objects
- Web-based monitoring dashboard

## Tech Stack
- Python
- YOLOv8 (Ultralytics)
- Flask
- HTML / CSS / JavaScript

## Project Structure

smart-surveillance  
│  
├── backend  
│   ├── app.py  
│   ├── detect.py  
│   └── yolov8n.pt  
│  
├── frontend  
│   ├── index.html  
│   ├── live.html  
│   ├── history.html  
│   ├── alerts.html  
│   ├── css/style.css  
│   └── js/script.js  
│  
├── run_project.bat  
└── .gitignore  
