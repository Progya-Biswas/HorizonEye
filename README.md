# HorizonEye 🚀
### Vision-Based Horizon Detection & Flight Altitude Estimation

HorizonEye is a computer vision-based flight perception system designed to detect the natural horizon from camera images and estimate aircraft orientation and altitude-related parameters.

The system processes visual information to identify the sky-ground boundary, extract horizon geometry, and derive flight-related measurements using image processing and geometric modeling.

## ✨ Key Features

- **Horizon Detection:** Identifies the sky-ground boundary using image segmentation and thresholding techniques.
- **Sky-Ground Segmentation:** Applies Otsu thresholding to distinguish between sky and ground regions.
- **Horizon Line Fitting:** Estimates the horizon line from segmented image regions.
- **Aircraft Roll Estimation:** Uses the detected horizon orientation to estimate aircraft roll.
- **Horizon Dip Angle:** Determines the angular displacement of the observed horizon.
- **Altitude Estimation:** Uses camera geometry and spherical-Earth geometry to estimate flight altitude.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Numerical Computing | NumPy |
| Image Processing | Otsu Thresholding, Image Segmentation |
| Geometry | Camera Geometry, Spherical-Earth Geometry |

## ⚙️ System Workflow

The HorizonEye pipeline follows these stages:

1. **Image Acquisition:** Capture or load visual input from a camera.
2. **Preprocessing:** Prepare the image for segmentation and horizon analysis.
3. **Sky-Ground Segmentation:** Separate sky and ground regions using Otsu thresholding.
4. **Horizon Detection:** Identify the boundary between the two regions.
5. **Horizon Line Fitting:** Fit a line to the detected horizon points.
6. **Flight Parameter Estimation:** Calculate aircraft roll and horizon dip angle.
7. **Altitude Estimation:** Apply camera calibration and spherical-Earth geometry to estimate flight altitude.

## 📐 Technical Approach

### 1. Sky-Ground Segmentation
Otsu thresholding is used to separate sky and ground regions based on image intensity distributions.

### 2. Horizon Line Fitting
The detected boundary points are used to estimate a horizon line, providing the geometric representation required for subsequent calculations.

### 3. Aircraft Roll Estimation
The orientation of the detected horizon line is used to estimate the aircraft's roll angle relative to the camera coordinate system.

### 4. Horizon Dip & Altitude Estimation
The horizon dip angle is used alongside camera parameters and spherical-Earth geometry to estimate aircraft altitude.

Accurate altitude estimation requires appropriate camera calibration, including field of view and camera orientation. Estimates generated without validated calibration should be treated as experimental rather than measured flight altitudes.

## 📊 Applications

- Aircraft visual perception systems
- Vision-based flight monitoring
- Autonomous aerial systems
- UAV navigation research
- Computer vision-based aerospace applications

## 🔬 Project Status

**Status:** Under Development

Current development focuses on improving horizon detection, refining geometric estimation, and evaluating the system under different visual and camera conditions.

## 🚀 Future Improvements

- Improve horizon detection under varying lighting and weather conditions.
- Implement robust horizon tracking across video frames.
- Integrate camera calibration for more reliable altitude estimation.
- Evaluate performance using different camera configurations.
- Explore integration with UAV and autonomous flight systems.

## 👩‍💻 Author

**Progya Paromita Biswas**

B.Tech Computer Science and Engineering  
Vellore Institute of Technology

---

*HorizonEye explores the intersection of computer vision, autonomous perception, and aerospace engineering to make visual flight analysis more accessible.*
