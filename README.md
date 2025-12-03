# ShotSense-AI

AI-powered cricket shot analysis & stance consistency scoring system
built using FastAPI, RTMPose & Random Forest models.

---

## 📌 Overview

ShotSense-AI is an intelligent cricket analytics system that evaluates
**two major components** of batting technique:

### ✅ 1. Intent-Based Shot Scoring (AI Model)

The user uploads a batting video and selects the intended shot type
from: **cut, drive, flick, misc, pull, slog, sweep**.

The backend: - Extracts frames from the video - Runs RTM-Pose to detect
body keypoints - Identifies the exact action frame using: - Rapid angle
change - Hand velocity - Converts the pose into feature vectors -
Classifies the shot using a **Random Forest model** - Produces an
AI-generated shot consistency score

### ✅ 2. Stance Consistency Tracker (Non-AI Logic)

The user uploads **5 stance videos**.

The system: - Extracts posture frames using RTM-Pose - Normalizes
keypoints - Calculates similarity between stances using **cosine
similarity** - Returns a final consistency score (0--100)

This module focuses purely on pre-shot stance --- not the stroke
execution.

---

## 🚀 Tech Stack

### **Backend**

- FastAPI (Python)
- RTM-Pose (MMPose)
- Random Forest (scikit-learn)
- OpenMMLab libraries
- FFmpeg (for video frame extraction)

### **Frontend (In Progress)**

- React Native (cross-platform mobile app)

---

## 📂 Project Structure (Simplified)

    project/
    │── main.py
    │── requirements.txt
    │── features/
    │   └── SHOT_CLASSIFICATION_SYSTEM/
    │       ├── rtmpose_models/
    │       ├── frame_extractor.py
    │       ├── pose_analysis.py
    │       ├── feature_builder.py
    │       ├── random_forest_model.pkl
    │       └── stance_consistency.py
    └── README.md

---

## 🛠️ Installation Guide (FastAPI Backend)

### **1. Install Python**

Download Python 3.8 - 3.10 from python.org\
➡️ Make sure to enable **"Add Python to PATH"**

Verify installation:

```bash
python --version
```

---

### **2. Create Project Folder & Enter It**

```bash
cd path/to/your/project/ShotSense-AI
```

---

### **3. Create a Virtual Environment**

```bash
python -m venv venv
```

### Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

---

### **4. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

### **5. Install OpenMMLab Libraries**

```bash
pip install openmim
pip install mmengine
mim install mmcv==2.1.0
mim install mmdet==3.2.0
mim install mmpose==1.2.0 --no-deps
```

---

### **6. Download RTM-Pose Model Files**

```bash
mkdir -p features/SHOT_CLASSIFICATION_SYSTEM/rtmpose_models
```

Download config:

```bash
wget -P features/SHOT_CLASSIFICATION_SYSTEM/rtmpose_models   https://github.com/open-mmlab/mmpose/tree/main/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py
```

Download checkpoint:

```bash
wget -P features/SHOT_CLASSIFICATION_SYSTEM/rtmpose_models   https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth
```

---

### **7. Dataset Setup & Model Training**

This project uses the "Cricket Shots IPL 2023" dataset.

**Step 1: Download**

Download the dataset manually from Kaggle:
[Cricket Shots IPL 2023](https://www.kaggle.com/taarunsridhar/cricket-shots-ipl-2023)

**Step 2: Extract**

Extract the dataset into the folder relative to the repo root:

```
CricAI-api/datasets/cricket-shots
```

The folder structure should look like:

```
SHOTSENSE-AI/datasets/cricket-shots/
├── cut/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── drive/
│   ├── video1.mp4
│   └── ...
├── flick/
├── misc/
├── pull/
├── slog/
└── sweep/
```

`config.py` already points to:

```python
DATASET_PATH = "datasets/cricket-shots"
```

**Step 3: Train the Model**

Once the dataset is ready, you can train the Random Forest model:

```bash
# Make sure you're in the project root
cd CricAI-api

# Run training
python features/SHOT_CLASSIFICATION_SYSTEM/model_training/train_model.py
```

After training, the model and scaler will be saved automatically to:

```
features/SHOT_CLASSIFICATION_SYSTEM/trained_models/
```

### **8. Run the FastAPI Server**

```bash
python main.py
```

If everything worked, you'll see:

    INFO:     Uvicorn running on http://0.0.0.0:8000
    INFO:     Application startup complete

Open Swagger Documentation: 👉 http://localhost:8000/docs

---

## 🔥 Available Endpoints

Method Endpoint Description

---

GET `/batting/shot-types` Get list of supported shots
POST `/batting/analyze-shot` Analyze a single batting video
POST `/batting/batch-analyze` Analyze multiple videos
GET `/batting/health` Health check

---

## 📱 Frontend (Coming Soon)

The mobile app will be built using **React Native**, with: - Video
upload\

- Real-time shot scoring display\
- Stance comparison dashboard\
- History & performance charts

This README will be updated once frontend development begins.

---

## 📄 License

MIT License © 2025 Yashodha Jayasinghe

---

## 🤝 Contributing

Pull requests are welcome!

---

## 📧 Contact

For questions or collaboration:\
**yashodhalasithjayasinghe@gmail.com**
