# Driver Drowsiness Detection 🚗
![made-with-python](https://img.shields.io/badge/made%20with-Python-blue.svg)

Real-time drowsiness detection system using MediaPipe facial landmarks and Eye Aspect Ratio (EAR) calculation to alert drivers when their eyes remain closed for extended periods.

## Overview
This system monitors drowsiness by detecting the driver's face and sending an audio alert if their eyes have been closed for a certain number of consecutive frames. The user can configure their own threshold by specifying an EAR ratio and customizing the number of frames through an intuitive GUI interface.

---

## Features
- ✨ Real-time face and eye detection using MediaPipe
- 🎯 Customizable EAR threshold and consecutive frame count
- 🔔 Audio alert system for drowsiness detection
- 🖥️ User-friendly GUI with settings panel
- 📊 Auto-calibration option for personalized thresholds
- 🎥 Multi-camera support

---

## Libraries Used
- `mediapipe` - Facial landmark detection
- `opencv-python` - Video processing and drawing
- `tkinter` - GUI interface
- `playsound` - Audio alert playback
- `scipy` - Euclidean distance calculations
- `imutils` - Video stream handling
- `numpy` - Array operations

---

## Installation

### Prerequisites
- Python 3.6 or higher
- Webcam/camera access

### Setup

1. **Clone the repository**
```bash
