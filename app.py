# app.py — Forward Crossing + Calibrated MPH + Near Miss Detection

import os, tempfile
from collections import deque

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

CAR_CLASS_ID = 2

YELLOW = (0, 255, 255)
GREEN  = (0, 255, 0)
RED    = (0, 0, 255)

# -------------------------------------------------------
# Utility
# -------------------------------------------------------

def make_writer(path, fps, w, h):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))

def bb_centroid(bb):
    x1, y1, x2, y2 = bb
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def build_homography(px_points, rect_w, rect_l):
    src = np.array(px_points, dtype=np.float32)
    dst = np.array([[0,0],[rect_w,0],[rect_w,rect_l],[0,rect_l]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)

def project_point(H, cx, cy):
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    gp = cv2.perspectiveTransform(pt, H)[0][0]
    return float(gp[0]), float(gp[1])

# -------------------------------------------------------
# Tracker
# -------------------------------------------------------

class Tracker:

    def __init__(self, fps, H=None):
        self.fps = fps
        self.H = H
        self.tracks = {}
        self.next_id = 1

    def update(self, dets, frame_idx):

        for d in dets:
            matched = False

            for tid, tr in self.tracks.items():

                # FIXED HERE (using tr["curr"] instead of tr["cx"])
                if np.hypot(d["cx"] - tr["curr"][0],
                            d["cy"] - tr["curr"][1]) < 60:

                    tr["bbox"] = d["bbox"]
                    tr["prev"] = tr["curr"]
                    tr["curr"] = (d["cx"], d["cy"])
                    tr["last"] = frame_idx

                    if self.H is not None:
                        gx, gy = project_point(self.H, d["cx"], d["cy"])
                        tr["hist"].append((frame_idx, gx, gy))

                    d["id"] = tid
                    matched = True
                    break

            if not matched:
                tid = self.next_id
                self.next_id += 1

                hist = deque(maxlen=60)
                if self.H is not None:
                    gx, gy = project_point(self.H, d["cx"], d["cy"])
                    hist.append((frame_idx, gx, gy))

                self.tracks[tid] = {
                    "bbox": d["bbox"],
                    "curr": (d["cx"], d["cy"]),
                    "prev": (d["cx"], d["cy"]),
                    "hist": hist,
                    "mph": None,
                    "last": frame_idx
                }

                d["id"] = tid

        return dets

# -------------------------------------------------------
# Analysis
# -------------------------------------------------------

def analyze(video_path, H,
            enable_nm,
            near_dist_m,
            near_ttc_s):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO("yolov8n.pt")
    tracker = Tracker(fps, H)

    out_path = "annotated_output.mp4"
    writer = make_writer(out_path, fps, w, h)

    near_events = []
    active_pairs = set()

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        dets = []

        if results.boxes is not None:
            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                if int(cls) != CAR_CLASS_ID:
                    continue

                bb = box.cpu().numpy()
                cx, cy = bb_centroid(bb)

                dets.append({
                    "bbox": bb,
                    "cx": cx,
                    "cy": cy
                })

        dets = tracker.update(dets, frame_idx)

        # -------- SPEED CALCULATION --------
        for tr in tracker.tracks.values():
            if H is not None and len(tr["hist"]) >= 2:

                f0, x0, y0 = tr["hist"][-2]
                f1, x1, y1 = tr["hist"][-1]

                dt = (f1 - f0) / fps
                if dt > 0:
                    dist = np.hypot(x1 - x0, y1 - y0)
                    tr["mph"] = (dist / dt) * 2.23694

        conflict_ids = set()

        # -------- NEAR MISS --------
        if enable_nm and H is not None:

            tracks = list(tracker.tracks.items())

            for i in range(len(tracks)):
                for j in range(i + 1, len(tracks)):

                    id1, t1 = tracks[i]
                    id2, t2 = tracks[j]

                    if len(t1["hist"]) < 2 or len(t2["hist"]) < 2:
                        continue

                    _, x1m, y1m = t1["hist"][-1]
                    _, x2m, y2m = t2["hist"][-1]

                    dx = x2m - x1m
                    dy = y2m - y1m
                    dist_m = np.hypot(dx, dy)

                    if dist_m > near_dist_m:
                        continue

                    v1 = np.array(t1["hist"][-1][1:]) - np.array(t1["hist"][-2][1:])
                    v2 = np.array(t2["hist"][-1][1:]) - np.array(t2["hist"][-2][1:])
                    rel_v = (v2 - v1) * fps

                    direction = np.array([dx, dy]) / (dist_m + 1e-6)
                    closing = np.dot(rel_v, direction)

                    if closing < 0:
                        ttc = dist_m / abs(closing)

                        if ttc <= near_ttc_s:

                            pair = tuple(sorted((id1, id2)))

                            if pair not in active_pairs:
                                near_events.append({
                                    "time_s": round(frame_idx / fps, 2),
                                    "car_1": id1,
                                    "car_2": id2,
                                    "distance_m": round(dist_m, 2),
                                    "ttc_s": round(ttc, 2),
                                    "speed_1_mph": t1["mph"],
                                    "speed_2_mph": t2["mph"]
                                })
                                active_pairs.add(pair)

                            conflict_ids.update([id1, id2])

        # -------- DRAW --------
        for tid, tr in tracker.tracks.items():

            x1, y1, x2, y2 = map(int, tr["bbox"])

            if tid in conflict_ids:
                color = RED
            else:
                color = YELLOW

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"ID {tid}"
            if tr["mph"] is not None:
                label += f" {tr['mph']:.1f} mph"

            cv2.putText(frame, label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    near_df = pd.DataFrame(near_events)
    return out_path, near_df

# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------

st.title("Traffic Analytics")

uploaded = st.file_uploader("Upload video",
                            type=["mp4","mov","avi","mkv"])

use_calib = st.checkbox("Enable Speed Calibration", value=False)
enable_nm = st.checkbox("Enable Near Miss Detection", value=True)

near_dist_m = st.slider("Near Miss Distance (m)", 1.0, 20.0, 5.0)
near_ttc_s  = st.slider("TTC Threshold (s)", 0.5, 5.0, 2.0)

run = st.button("Run")

if run and uploaded:

    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, uploaded.name)

    with open(video_path, "wb") as f:
        f.write(uploaded.getbuffer())

    H = None

    if use_calib:
        px_points = [(100,100),(300,100),(300,300),(100,300)]
        H = build_homography(px_points, 3.7, 10.0)

    out_path, near_df = analyze(video_path, H,
                                 enable_nm,
                                 near_dist_m,
                                 near_ttc_s)

    st.video(out_path)

    if not near_df.empty:
        st.subheader("Near Miss Events")
        st.dataframe(near_df)
    else:
        st.success("No near miss events detected.")
