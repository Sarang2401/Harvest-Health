# 🌾 Harvest Health - Crop Health Monitoring System

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Accuracy](https://img.shields.io/badge/ML%20Accuracy-98%25-brightgreen)
![Platform](https://img.shields.io/badge/platform-IoT%20%7C%20ML%20%7C%20Cloud-orange)

A smart and affordable solution for **real-time crop health monitoring** using IoT sensors and machine learning. Developed as a PBL (Project-Based Learning) initiative, this system helps farmers make informed decisions through data-driven insights.

---

## 🚀 Project Overview

Traditional farming methods rely on manual crop checks, which are time-consuming and error-prone. **Harvest Health** replaces guesswork with:

- 📶 **IoT Sensors** that gather real-time environmental and soil data  
- 🧠 **ML-powered Leaf Disease Detection**  
- ☁️ **Cloud Storage** with ThingSpeak  
- 📱 **User Interface** for visualization and alerts *(front-end editable by you)*

---

## 🎯 Problem We Solved

- ❌ Inefficient manual health monitoring  
- ❌ Delayed disease detection  
- ❌ Difficult access to real-time data  
- ✅ **Affordable**, **Scalable**, and **Smart** alternative for farmers  

---

## 🛠️ Technologies & Skills Used

- **Embedded Systems**: Arduino IDE, ESP32  
- **Cloud & IoT**: ThingSpeak  
- **Machine Learning**: TensorFlow, CNN, Image Augmentation  
- **Web/App Development**: HTML/CSS *(plug in your code here)*  
- **Design**: CAD for Rover, Circuit Schematics  

---

## 📁 Repository Structure

| Folder         | Description |
|----------------|-------------|
| `Cloud`        | ESP32 `.ino` code to send sensor data to ThingSpeak |
| `Design`       | 3D Rover design and circuit schematic |
| `Embedded`     | `rover_code.ino` & `moisture_servo.ino` for data collection & mobility |
| `ML`           | Potato leaf disease dataset |
| `plant_app`    | ML model code and front-end app *(plug in your UI later)* |
| `README.md`    | 📘 This file |

---

## 🌐 System Architecture

```plaintext
[ Sensors ] → [ ESP32 ] → [ ThingSpeak Cloud ] → [ Data Dashboard ]
                                            ↘
                                       [ ML Model ]
                                            ↘
                                      [ Farmer's App ]
```

---

## 📊 Live Sensor Monitoring (ThingSpeak)

---

## 🧠 ML Model - Potato Disease Classification

🥔 2152 Images  
🎯 Categories: Healthy, Early Blight, Late Blight  
📈 Accuracy: **98%**  
🧠 Model: CNN with Conv2D, MaxPooling, Dropout layers

📷 **Training Accuracy/Loss Plot** 

---

## 🧪 How It Works

1. **📥 Data Collection**: Sensors measure temperature, humidity, light, and soil moisture.  
2. **📊 Initial Processing**: ESP32 filters raw data using thresholds.  
3. **☁️ Data Upload**: Sends to ThingSpeak for live viewing.  
4. **🧠 ML Prediction**: ML model classifies potato leaf disease via images.  
5. **📱 Visualization**: User-friendly front-end (you can insert your code/UI here).

---

## 🗺️ Future Scope

- 🌦️ Integrate rainfall & weather API  
- 🌱 Automate irrigation/digging system  
- 🔐 Secure MQTT data transmission  
- 🛠️ Add alert system via app  

---

## 👨‍💻 Team Members

- Aryan Suryawanshi  
- Atharva Kanawade  
- Sarang Shigwan  
- Aditya Shinde  
- Rutika Shekokar  
- Aarav Thigale  

---

## 📸 Gallery (Add screenshots if you have them)

- 🛞 Rover Design  
- 🧪 ML Confusion Matrix  
- 📊 Real-time ThingSpeak dashboard  
  

---

## 📚 References

- [ThingSpeak](https://thingspeak.com/)  
- [TensorFlow Image Classification Guide](https://www.tensorflow.org/tutorials/images/classification)  
- Public Dataset: Potato Leaf Disease

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

Made with 💚 by Team **Harvest Health**
