from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import numpy as np
import base64
import cv2
import math
from typing import Any

app = FastAPI(title="ETA Pose Analysis Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = none

LANDMARK_NAMES = {
    0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist", 17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index", 21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle", 29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index",
}

def angle_between(a, b, c):
    try:
        ba = [a[0] - b[0], a[1] - b[1]]
        bc = [c[0] - b[0], c[1] - b[1]]
        cosine = (ba[0]*bc[0] + ba[1]*bc[1]) / (
            math.sqrt(ba[0]**2 + ba[1]**2) * math.sqrt(bc[0]**2 + bc[1]**2) + 1e-6
        )
        return round(math.degrees(math.acos(max(-1, min(1, cosine)))), 1)
    except:
        return None

def extract_sprint_metrics(landmarks, image_width, image_height):
    def lm(idx):
        l = landmarks[idx]
        return [l.x * image_width, l.y * image_height] if l.visibility > 0.5 else None

    metrics = {}

    l_hip = lm(23); l_knee = lm(25); l_ankle = lm(27)
    r_hip = lm(24); r_knee = lm(26); r_ankle = lm(28)
    l_shoulder = lm(11); r_shoulder = lm(12)
    l_elbow = lm(13); r_elbow = lm(14)
    l_wrist = lm(15); r_wrist = lm(16)

    if l_hip and l_knee and l_ankle:
        metrics["left_knee_angle"] = angle_between(l_hip, l_knee, l_ankle)
    if r_hip and r_knee and r_ankle:
        metrics["right_knee_angle"] = angle_between(r_hip, r_knee, r_ankle)
    if l_shoulder and l_elbow and l_wrist:
        metrics["left_elbow_angle"] = angle_between(l_shoulder, l_elbow, l_wrist)
    if r_shoulder and r_elbow and r_wrist:
        metrics["right_elbow_angle"] = angle_between(r_shoulder, r_elbow, r_wrist)

    if l_hip and r_hip:
        avg_hip_y = (l_hip[1] + r_hip[1]) / 2
        metrics["hip_height_normalized"] = round(1 - (avg_hip_y / image_height), 3)

    if l_shoulder and r_shoulder and l_hip and r_hip:
        mid_shoulder = [(l_shoulder[0]+r_shoulder[0])/2, (l_shoulder[1]+r_shoulder[1])/2]
        mid_hip = [(l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2]
        dx = mid_shoulder[0] - mid_hip[0]
        dy = mid_hip[1] - mid_shoulder[1]
        metrics["trunk_lean_angle"] = round(math.degrees(math.atan2(dx, dy + 1e-6)), 1)

    l_foot = lm(31); r_foot = lm(32)
    if l_foot and r_foot:
        metrics["stride_width_px"] = round(abs(l_foot[0] - r_foot[0]), 1)

    if l_elbow and r_elbow and l_hip and r_hip:
        mid_hip_y = (l_hip[1] + r_hip[1]) / 2
        l_arm_drive = mid_hip_y - l_elbow[1]
        r_arm_drive = mid_hip_y - r_elbow[1]
        if l_arm_drive + r_arm_drive > 0:
            metrics["arm_symmetry"] = round(
                min(l_arm_drive, r_arm_drive) / max(l_arm_drive, r_arm_drive + 1e-6), 3
            )

    if l_knee and r_knee and l_hip and r_hip:
        mid_hip_y = (l_hip[1] + r_hip[1]) / 2
        higher_knee_y = min(l_knee[1], r_knee[1])
        metrics["knee_drive_ratio"] = round((mid_hip_y - higher_knee_y) / (image_height + 1e-6), 3)

    return metrics

def compute_form_score(metrics):
    score = 60

    for side in ["left_knee_angle", "right_knee_angle"]:
        angle = metrics.get(side)
        if angle:
            if 85 <= angle <= 135:
                score += 5
            elif angle < 70 or angle > 160:
                score -= 5

    hip_h = metrics.get("hip_height_normalized", 0)
    if hip_h > 0.55:
        score += 8
    elif hip_h < 0.4:
        score -= 8

    lean = metrics.get("trunk_lean_angle")
    if lean is not None:
        if 5 <= lean <= 20:
            score += 6
        elif lean > 30 or lean < -5:
            score -= 6

    arm_sym = metrics.get("arm_symmetry")
    if arm_sym is not None:
        if arm_sym > 0.8:
            score += 5
        elif arm_sym < 0.5:
            score -= 5

    knee_drive = metrics.get("knee_drive_ratio", 0)
    if knee_drive > 0.08:
        score += 6
    elif knee_drive < 0.02:
        score -= 4

    return max(0, min(100, round(score)))

@app.get("/health")
def health():
    return {"status": "ok", "service": "ETA Pose Analysis"}

@app.post("/analyze")
async def analyze(req: dict[str, Any]):
    try:
        image_base64 = req.get("image_base64", "")
        if not image_base64:
            raise HTTPException(status_code=400, detail="image_base64 is required")

        img_data = base64.b64decode(image_base64)
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        ) as pose:
            results = pose.process(image_rgb)

        if not results.pose_landmarks:
            return {
                "success": False,
                "keypoints": {},
                "metrics": {},
                "form_score": 0,
                "error": "No pose detected. Ensure the athlete's full body is visible."
            }

        landmarks = results.pose_landmarks.landmark
        keypoints = {}
        for idx, name in LANDMARK_NAMES.items():
            lm = landmarks[idx]
            keypoints[name] = {
                "x": round(lm.x, 4),
                "y": round(lm.y, 4),
                "z": round(lm.z, 4),
                "visibility": round(lm.visibility, 3),
            }

        metrics = extract_sprint_metrics(landmarks, w, h)
        form_score = compute_form_score(metrics)

        return {
            "success": True,
            "keypoints": keypoints,
            "metrics": metrics,
            "form_score": form_score,
            "error": None
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
