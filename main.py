import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """ Calculate angle between three points """
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def classify_pose(landmarks, image_height, image_width):
    def get_point(name):
        lm = landmarks[mp_pose.PoseLandmark[name].value]
        return [lm.x, lm.y]

    # Get keypoints
    left_hip = get_point('LEFT_HIP')
    right_hip = get_point('RIGHT_HIP')
    left_knee = get_point('LEFT_KNEE')
    right_knee = get_point('RIGHT_KNEE')
    left_ankle = get_point('LEFT_ANKLE')
    right_ankle = get_point('RIGHT_ANKLE')
    left_shoulder = get_point('LEFT_SHOULDER')
    right_shoulder = get_point('RIGHT_SHOULDER')

    # Midpoints
    mid_hip = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
    mid_knee = [(left_knee[0] + right_knee[0]) / 2, (left_knee[1] + right_knee[1]) / 2]
    mid_ankle = [(left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2]
    mid_shoulder = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]

    # Bounding box
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    box_width = (max(xs) - min(xs)) * image_width
    box_height = (max(ys) - min(ys)) * image_height

    # Distances
    shoulder_to_hip = abs(mid_shoulder[1] - mid_hip[1]) * image_height
    hip_to_ankle_v = abs(mid_hip[1] - mid_ankle[1]) * image_height
    hip_to_knee_v = abs(mid_hip[1] - mid_knee[1]) * image_height
    hip_to_knee_h = abs(mid_hip[0] - mid_knee[0]) * image_width
    hip_to_ankle_h = abs(mid_hip[0] - mid_ankle[0]) * image_width
    body_height = abs(mid_shoulder[1] - mid_ankle[1]) * image_height

    # ----- Classification -----
    if box_width > box_height * 1.5 or body_height < image_height * 0.2:
        return "Lying"

    # Side-view sitting
    if hip_to_knee_v < image_height * 0.15 and hip_to_knee_h > image_width * 0.08:
        return "Sitting"

    # Front-view sitting (folded legs, small vertical height)
    if hip_to_ankle_v < image_height * 0.2 and hip_to_knee_v < image_height * 0.2:
        return "Sitting"

    # Backup: medium body height, hip clearly below shoulder
    if image_height * 0.2 < body_height < image_height * 0.6 and mid_hip[1] > mid_shoulder[1]:
        return "Sitting"

    # Tall body → Standing
    if body_height > image_height * 0.6:
        return "Standing"

    return "Unknown"

# ------------- MAIN ----------------

# Load your image
image_path = 'standing.jpg'  # <<< CHANGE to your image path
image = cv2.imread(image_path)
image_height, image_width, _ = image.shape

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process image
results = pose.process(image_rgb)

if results.pose_landmarks:
    # Draw pose landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Classify pose
    label = classify_pose(results.pose_landmarks.landmark, image_height)
    print(f"Detected Pose: {label}")

    # Put label on image
    cv2.putText(image, label, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
else:
    print("No person detected!")

# Show the output
cv2.imshow('Pose Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
