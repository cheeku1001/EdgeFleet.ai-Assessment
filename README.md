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

1. Clone the repository:
   git clone https://github.com/YourUsername/cricket-ball-trajectory-hybrid.git

2. Install dependencies:
   pip install -r requirements.txt

3. Place your test videos in a folder:
   /content/drive/MyDrive/cricket_videos/

4. Update `VIDEO_DIR` inside `code/ball_tracking_dataset.py` with your folder path

5. Run the tracker:
   python code/ball_tracking_dataset.py

6. Outputs:
   - `conclusion/<video_name>/trajectory.mp4`
   - `conclusion/<video_name>/ball_coordinates.csv`

