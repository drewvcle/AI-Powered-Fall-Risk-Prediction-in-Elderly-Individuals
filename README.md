# AI-Powered-Fall-Risk-Prediction-in-Elderly-Individuals

This capstone project is currently ongoing. Estimated completion date is March 2026.

This repository contains the implementation of _AI Powered Fall Risk Prediction in Elderly Individuals_. This is accomplished by using a wearable inertial measurement unit (IMU). This project emphasizes early fall risk prediction of 3-5 seconds by identifying pre-fall motions prior to an actual fall event.

This implementation utilizes **LIMU-BERT**, a deep learning model specifically designed for inertial time-series data. **MobiFall** dataset was utilized to train the deep learning model.
Other datasets that are worth mentioning are **SisFall** and **FARSEEING**, and both were used in the initial phase of testing.

> **Project Status**: WIP.

## Project Objectives
The primary objective of this engineering design project is to create a functional, real-time system capable of predicting fall risk for elderly people by using wearable IMU sensor data and deep learning. Based on this goal, the project targets several technical and system level objectives:
1. Design and implement a wearable sensing platform equipped with an IMU and a microcontroller capable of continuous capture of motion-related data from the user.
2. Design a preprocessing pipeline to clean, segment, and normalize the IMU signals. The data is prepared for training the deep learning AI model and real-time inference.
3. Train and AI model that can evaluate signals to determine its gait instability, irregular motion patterns, and any outliers for potential fall risks with high reliability.
4. The trained AI model will be deployed on the wearable microcontroller to enable on-device inference for low latency real-time prediction.
5. Implementation of bluetooth communication to send sensor readings and/or prediction results to the connected mobile device.
6. Design a mobile application interface to display user status and receive notifications of fall risk while sending data to the backend server.
7. Develop a backend cloud service to store user data, manage alerts, and allow communication between the wearable device and mobile application.
8. Expand upon a Minimum Viable Product that achieves a seamless integration between the hardware, software, and the networking components between platforms.
9. Meet real-time constraints, targeting an alert latency of under six seconds from the moment a fall risk is detected.
10. Support user notification functionality through alerts delivered via the mobile application.

Thus, the problem definition is broken down into four components:
1. Preprocess data for model intake
2. Develop, train, and deploy AI model
3. Develop mobile application for notifications
4. Setup hardware and cloud infrastructure, system integration, and final testing

## Preprocessing Approach
- Microcontroller with IMU unit will output 6 signals: accelerometer signals in 3 axes + gyroscope signals in 3 axes
- Compute resampling of dataset to a fixed frequency
- Clean noise and sensor drift with filters (low-pass Butterworth filter to remove high frequency noise)
- Explicit PRE_FALL labeling to prior fall events
- Segment sliding windows of time-series data
- Label windows

---

## Label Definitions

| Label     | Description |
|----------|-------------|
| NON      | Normal activity |
| PRE_FALL | Pre-fall motion patterns indicating elevated fall risk |
| FALL     | Actual fall event |


