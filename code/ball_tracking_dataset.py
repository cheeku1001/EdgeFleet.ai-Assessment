import cv2
import numpy as np
import pandas as pd
import os


VIDEO_DIR = "/content/drive/MyDrive/cricket_videos"
OUTPUT_DIR = "outputs"


os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- COLOR THRESHOLDS  ---
lower_white = np.array([80, 30, 100])
upper_white = np.array([150, 50, 255])
lower_red = np.array([0, 30, 100])
upper_red = np.array([20, 90, 255])

video_files = [v for v in os.listdir(VIDEO_DIR) if v.lower().endswith((".mp4", ".mov"))]
print(f"Found {len(video_files)} videos")

for video_name in video_files:
    print(f"\nProcessing: {video_name}")

    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, video_name))
    backSub = cv2.createBackgroundSubtractorMOG2(100, 50, False)

    last_ball_pos = None
    frame_id = 0

    raw_points = []     # for regression
    csv_rows = []       # final CSV

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    name = os.path.splitext(video_name)[0]
    out_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    out = cv2.VideoWriter(
        os.path.join(out_dir, "annotated_video.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h)
    )

    # =========================
    # 1️⃣ DETECTION PASS
    # =========================
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        color_mask = cv2.bitwise_or(mask_white, mask_red)

        fg_mask = backSub.apply(frame)
        mask = cv2.bitwise_and(color_mask, fg_mask)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            circularity = 4*np.pi*area/(peri**2)
            if 15 < area < 250 and circularity > 0.4:
                (x,y),r = cv2.minEnclosingCircle(cnt)
                candidates.append(((int(x),int(y)), int(r)))

        visible = 0
        cx = cy = r = None

        if candidates:
            if last_ball_pos is None:
                best = candidates[0]
            else:
                best = min(
                    candidates,
                    key=lambda c: np.linalg.norm(
                        np.array(c[0]) - np.array(last_ball_pos)
                    )
                )
            cx, cy = best[0]
            r = best[1]
            visible = 1
            last_ball_pos = (cx, cy)
            raw_points.append((frame_id, cx, cy))

            cv2.circle(output, (cx, cy), r, (0,255,0), 2)
            cv2.circle(output, (cx, cy), 3, (0,255,0), -1)

        csv_rows.append([frame_id, cx, cy, r, visible])
        out.write(output)
        frame_id += 1

    cap.release()
    out.release()

    # =========================
    # 2️⃣ LINEAR REGRESSION TRAINING
    # =========================
    if len(raw_points) > 10:
        data = np.array(raw_points)
        t = data[:,0]
        x = data[:,1]
        y = data[:,2]

        px = np.polyfit(t, x, 1)     # linear
        py = np.polyfit(t, y, 2)     # parabola

        t_fit = np.arange(t.min(), t.max())
        x_fit = np.polyval(px, t_fit)
        y_fit = np.polyval(py, t_fit)

        traj_df = pd.DataFrame({
            "frame": t_fit,
            "x_fit": x_fit.astype(int),
            "y_fit": y_fit.astype(int)
        })
        traj_df.to_csv(os.path.join(out_dir, "trajectory_fitted.csv"), index=False)

    # =========================
    # 3️⃣ SAVE RAW CSV
    # =========================
    pd.DataFrame(
        csv_rows,
        columns=["frame","x","y","radius","visible"]
    ).to_csv(os.path.join(out_dir, "ball_coordinates.csv"), index=False)

    print(f"Saved outputs to {out_dir}")

print("\n✅ ALL VIDEOS PROCESSED")
