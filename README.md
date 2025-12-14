# 🎭 AI-Based Facial Emotion Detection System (FER2013)

## 📌 Project Overview
Traditional digital learning platforms track clicks, time-on-task, and completion rates, but completely ignore emotional engagement. This project addresses that blind spot by building an AI-powered facial emotion recognition system that detects learner emotions in real time using webcam input.

The system classifies facial expressions into seven core emotions and provides live analytics, enabling emotion-aware interventions in online learning environments.

This repository contains the complete end-to-end pipeline: data preparation, model training, evaluation, and real-time deployment.

## 🎯 Problem Statement
Current e-learning platforms fail because they:

- Cannot detect confusion or frustration
- Miss boredom and disengagement
- Ignore positive or negative emotional reactions
- Fail to notice attention drops, leading to higher dropout rates

### Bottom line: Learning systems are emotionally blind.

## 💡 Proposed Solution
### An AI-driven facial emotion recognition engine that:

- Analyzes real-time webcam video
- Converts facial expressions into actionable emotional insights
- Enables immediate pedagogical intervention
- Closes the emotional feedback loop to improve retention

Pipeline:
### Webcam Input → CNN-based Emotion Model → Live Analytics Dashboard

## 😐 Supported Emotion Classes
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## ⚙️ Technical Approach
### Core Technologies
- Facial Landmark Detection for feature extraction
- Convolutional Neural Networks (CNNs) for emotion classification
- TensorFlow / PyTorch for deep learning workflows
  
`` Reality check: Facial recognition is not emotion detection.
  Emotion detection depends on facial recognition, but they are not the same problem. ``

## 📊 Dataset
### Source
- FER2013 Dataset (Kaggle)
  https://www.kaggle.com/datasets/msambare/fer2013

### Dataset Statistics
| Emotion  | Images |
| -------- | ------ |
| Angry    | 3,995  |
| Disgust  | 436    |
| Fear     | 4,097  |
| Happy    | 7,215  |
| Sad      | 4,830  |
| Surprise | 3,171  |
| Neutral  | 4,965  |

- Training set: 28,709 images
- Test set: 7,178 images
- Total: 35,887 images

## ⚠️ Data Quality Issues Identified
These are real limitations, not academic excuses:
- ~15% low-quality samples (blur, occlusion, poor lighting)
- ~8% near-duplicate images
- High label inconsistency (ambiguous expressions)
- Distribution imbalance (underrepresented emotions like disgust & fear)
Ignoring these issues would invalidate results. They were explicitly audited and handled.

## 🧹 Data Preparation
### Preprocessing Pipeline
- Image normalization & resizing
- Label standardization
- Dataset structure unification
- Version-controlled preprocessing steps
### Drift Handling
#### Problem:
New data introduced variations in lighting, pose, camera quality, and demographics → silent accuracy degradation.
#### Solution:
- Automated input drift detection
- Emotion distribution monitoring
- Label consistency checks
- Retraining triggers when thresholds are exceeded

## 🧠 Model Training
### Training Details
- Architecture: CNN
- Epochs: 50
- Training Time: ~3.45 hours (GPU)
- Best Epoch: 38

### Achieved Performance
- Validation Accuracy: 69.23%
- Validation Loss: 0.8234

FER2013 benchmark reality:
  - Human performance: ~65%
  - Strong academic models: 65–70%
  - State-of-the-art ceiling: ~73%
This model is competitive, not magical.

## 🧪 Evaluation
Evaluation includes:
- Classification report (precision, recall, F1)
- Confusion matrix
- Training & validation loss curves
- Per-class performance breakdown
#### Target threshold: >60% test accuracy
#### Achieved: ~65–70%

## 🎥 Real-Time Emotion Detection
### Features
- Live webcam face detection
- Emotion label + confidence score
- Real-time emotion distribution
- Short-term emotion trend analysis
- FPS performance counter
- Exportable analytics

### Run Live System
``` python realtime_emotion_analytics.py ```

## 📁 Project Structure
```
├── prepare_fer2013_data.py
├── train_model.py
├── test_model.py
├── realtime_emotion_analytics.py
├── face_emotion_system.py
├── data/
│   └── fer2013/                   # Download from Kaggle
│       ├── train/
│       │   ├── angry/
│       │   ├── disgust/
│       │   ├── fear/
│       │   ├── happy/
│       │   ├── sad/
│       │   ├── surprise/
│       │   └── neutral/
│       └── test/
│           └── (same structure)
├── models/
├── logs/
└── results/

```
## 🎯 Complete Workflow
 ``` Step 1: Prepare Data(5 minutes) → Step 2: Train Model(2-4 hours GPU) → Step 3: Test Model(10 minutes) → Step 4: Live Testing(Real-time)```

## 🚀 STEP-BY-STEP EXECUTION
⬇️ Prerequisites (One-time Setup)
#### 1. Install all dependencies
pip install opencv-python tensorflow numpy matplotlib pandas scikit-learn seaborn

#### 2. Download FER2013 dataset from Kaggle
#### Visit: https://www.kaggle.com/datasets/msambare/fer2013
#### Download and extract to: data/fer2013/

#### 3. Verify structure
- data/fer2013/train/angry/    # Should have images
- data/fer2013/train/happy/    # Should have images

## 📊 STEP 1: Prepare Data (5 minutes)
### What it does:
- ✅ Validates dataset structure
- ✅ Analyzes class distribution
- ✅ Visualizes sample images
- ✅ Calculates class weights
- ✅ Generates preparation report
Run:
**python prepare_fer2013_data.py**
Expected output:

Step 1: Validating dataset structure...
 - ✓ Found train directory: data/fer2013/train
    - ✓ angry     :  3995 images
    - ✓ disgust   :   436 images
    - ✓ fear      :  4097 images
    - ✓ happy     :  7215 images
    - ✓ sad       :  4830 images
    - ✓ surprise  :  3171 images
    - ✓ neutral   :  4965 images

- Training set: 28,709 images
- Test set: 7,178 images
- Total: 35,887 images

✓ DATA PREPARATION COMPLETE!

**Files created:**
- results/data_analysis/class_distribution.png
- results/data_analysis/sample_images.png
- results/data_analysis/class_weights.txt
- results/data_analysis/data_preparation_report.txt

## 🏋️ STEP 2: Train Model (2-4 hours on GPU)
What it does:
- ✅ Checks GPU availability
- ✅ Builds CNN architecture
- ✅ Applies data augmentation
- ✅ Trains for 50 epochs
- ✅ Saves best model
- ✅ Generates training plots

Run:
python train_model.py

Expected output:
 **FER2013 MODEL TRAINING PIPELINE**

## 📈 STEP 3: Test Model (10 minutes)
What it does:
- ✅ Loads trained model
- ✅ Evaluates on test set
- ✅ Generates confusion matrix
- ✅ Calculates per-class metrics
- ✅ Performs error analysis
- ✅ Creates evaluation report

Run:
python test_model.py

Expected output:
  **MODEL TESTING AND EVALUATION**

## 🎥 STEP 4: Live Testing (Real-time)
What it does:
- ✅ Loads trained model
- ✅ Runs real-time webcam detection
- ✅ Shows live statistics
- ✅ Tracks emotion distribution
- ✅ Exports analytics data
  
Run:
python realtime_emotion_analytics.py

What you'll see:
Live video with:
- Face detection boxes
- Emotion labels with confidence
- Real-time statistics panel
- Emotion distribution bars
- Recent trend analysis (last 10 seconds)
- FPS counter

Controls:
  - 'q' - Quit and save statistics
  - 's' - Save screenshot
  - 'e' - Export data to CSV
  - 'r' - Reset statistics
  - SPACE - Pause/Resume

After testing (press 'q'):
  **SESSION SUMMARY**

## 📝 Quick Commands Cheat Sheet
- python prepare_fer2013_data.py    # 5 minutes
- python train_model.py              # 2-4 hours ☕☕☕
- python test_model.py               # 10 minutes
- python realtime_emotion_analytics.py  # Real-time

## 🚀 Deployment Checklist
- ✅ Model training completes successfully
- ✅ Test accuracy ≥ 60%
- ✅ Real-time webcam inference works
- ✅ FPS is usable for live scenarios

## ⚖️ Ethical & Legal Considerations
This system processes biometric facial data.

Key risks:
- Privacy violations
- Surveillance misuse
- Bias across demographics

Compliance requirement:
Must adhere to GDPR and responsible AI guidelines.
` Emotion detection without ethics is not innovation — it’s liability.`

## 🧾 Summary
This project delivers a practical, explainable, and benchmark-aligned AI emotion detection system using FER2013. It demonstrates:
- End-to-end ML pipeline design
- Real-world dataset limitations handling
- Competitive accuracy within known benchmarks
- Real-time deployment capability

No exaggerated claims. No fake “99% accuracy”.
Just solid engineering.

## 🙏 Acknowledgments
 <a href="https://www.python.org" target="_blank" rel="noopener noreferrer">
     🏗️ Built with
   </a>
, 
   <a href="https://www.kaggle.com/datasets/msambare/fer2013" target="_blank" rel="noopener noreferrer">
      ☁️ Data Source
   </a>

📧 Contact : For questions or feedback, please open an issue on GitHub or contact the maintainer.

## © License

This project is licensed under the MIT License - see the LICENSE file for details.
