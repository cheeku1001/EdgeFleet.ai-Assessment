# 🏏 Cricket Ball Trajectory Detection (Hybrid CV + Deep Learning)

This project detects and tracks a cricket ball from match videos using a hybrid
approach combining classical computer vision and deep learning (YOLOv8).

## 🔥 Features
- Ball detection using HSV + motion filtering
- YOLOv8 verification for robustness
- Frame-wise centroid tracking
- Trajectory overlay video
- CSV annotation per video

## Inputs in this drive 
-https://drive.google.com/file/d/1hnaGuqGuMXaFKI5fhfy8gatzCH-6iMcJ/
view?usp=sharing

## Outputs are in the form of
Outputs/
├── video_no./
│ ├── trajectory.mp4
│ └── ball_coordinates.csv

## ▶️ How to Run

```bash
pip install -r requirements.txt
python code/ball_tracking_dataset.py
