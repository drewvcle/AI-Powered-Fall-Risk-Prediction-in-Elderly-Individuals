# AI-Powered-Fall-Risk-Prediction-in-Elderly-Individuals

This capstone project is currently ongoing. Estimated completion date is March 2026.

This repository contains the implementation of _AI Powered Fall Risk Prediction in Elderly Individuals_. This project leverages a wearable inertial measurement unit (IMU) to collect motion data for early fall risk predction. If any pre-fall movement patterns are detected 3-5 before the fall, the sensor data is transmitted to the user's mobile device using Bluetooth Low Energy (BLE) and subsequently be sent to a backend service for AI inference. If an elevated risk is detected, the system will issue a warning notification to the user.

This implementation utilizes **LIMU-BERT**, a deep learning model specifically designed for inertial time-series data.
The MobiFall dataset was utilized to train the deep learning model.
Other datasets worth mentioning are **SisFall** and **FARSEEING**, both used in the initial phase of testing.

> **Project Status**: WIP.

---
## Figures from Milestone Report


<p align="center">
  [![](https://cdn.discordapp.com/attachments/771051131035189253/1450944209346494494/image18.png?ex=69446098&is=69430f18&hm=b4210747346f09726e3bf3906a365c183ef5e934dde986b92cd54dee994a66fb&)]()
  <sub><b>Figure 1.0.</b> Window Diagram of IMU signals.</sub>
</p>


[![](https://cdn.discordapp.com/attachments/771051131035189253/1450944209690300507/image14.png?ex=69446098&is=69430f18&hm=cef29e38b4d736ef710d72066e319e84b829c18a1301209e916352f17f951062&)]()
<sub>**Figure 1.1.** RAW IMU Signals (Accelerometer and Gyroscope).</sub>

[![](https://cdn.discordapp.com/attachments/771051131035189253/1450943988294090792/image23.png?ex=69446064&is=69430ee4&hm=19bf36c8bd99c7c5fea9695745ded2cf75729f349fac93e0f7b0cad8ac949776&)]()
<sub>**Figure 1.2.** Downsampling Comparison between Various Frequencies.</sub>

[![](https://cdn.discordapp.com/attachments/771051131035189253/1450943987564417146/image25.png?ex=69446063&is=69430ee3&hm=05bd8a14c3db21bd88ddaf4f7fffd0636e9710fb7ddb372d142d746fb512ee83&)]()
<sub>**Figure 2.0.** Proposed High-Level Diagram.</sub>

[![](https://cdn.discordapp.com/attachments/771051131035189253/1450943987983843458/image24.png?ex=69446063&is=69430ee3&hm=132acb38f1716fb238410082fea94771be3f4cde146c1364b47c042b5b804441&)]()
<sub>**Figure 2.1** Proposed Backend Architecture..</sub>

[![](https://cdn.discordapp.com/attachments/771051131035189253/1450943986847186955/image26.png?ex=69446063&is=69430ee3&hm=4765268c14451aa37e5912538fdcae4d2794d6cc91a9c9229bf382cc8f67fcb2&)]()
<sub>**Figure 3.0** Proposed Login Screen.</sub>

[![](https://cdn.discordapp.com/attachments/771051131035189253/1450943988969504900/image19.png?ex=69446064&is=69430ee4&hm=395a63250cc668d33408b500e726b8d66c74c0e1daa88be5a723f49498cc7951&)]()
[![](https://cdn.discordapp.com/attachments/771051131035189253/1450943987149045864/image27.png?ex=69446063&is=69430ee3&hm=fcde9924143e6184e6a066187ded168dc5385bd9c265cf76f2bd6cc065b6bba0&)]()
[![](https://cdn.discordapp.com/attachments/771051131035189253/1450943988616925248/image20.png?ex=69446064&is=69430ee4&hm=f0e1ff95e37a68944ae6d577c1f5a3b8ebb0c34dc16c85acc7ee0d740a59e3ce&)]()
[![](https://cdn.discordapp.com/attachments/771051131035189253/1450943988969504900/image19.png?ex=69446064&is=69430ee4&hm=395a63250cc668d33408b500e726b8d66c74c0e1daa88be5a723f49498cc7951&)]()
<sub>**Figure 3.1** Risk Related Mockup Modules.</sub>

---

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






