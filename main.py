from ultralytics import YOLO
import cv2
import time
from congestion_calculator import calculate_congestion, get_priority_road
import numpy as np

# Load model
model = YOLO("yolov8n.pt")

# Vehicle classes
vehicle_map = {2: "car", 3: "bike", 5: "bus", 7: "truck"}

# 4 Roads (videos or cameras)
caps = {
    "Road A": cv2.VideoCapture("roadA.mp4"),
    "Road B": cv2.VideoCapture("roadB.mp4"),
    "Road C": cv2.VideoCapture("roadC.mp4"),
    "Road D": cv2.VideoCapture("roadD.mp4")
}

# Lane sizes
lanes = {
    "Road A": None,
    "Road B": None,
    "Road C": None,
    "Road D": None
}

# Storage
prev_frames = {}
last_capture_time = {}
cycle_data = {}
was_waiting = {}
completed_roads = {}

MOTION_THRESHOLD = 50000


def detect_lanes(frame, road_name):

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blur, 50, 150)

    # Detect lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=120,
        maxLineGap=40
    )

    lane_lines = []

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line[0]

            # Keep mostly vertical lines
            if abs(x1 - x2) < 50 and abs(y2 - y1) > 100:

                lane_lines.append(line)

                # Draw detected line (for debugging)
                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

    # Estimate number of lanes
    num_lanes = max(1, min(len(lane_lines) - 1, 6))

    print(f"{road_name} detected lanes:", num_lanes)

    return num_lanes


def is_traffic_stopped(prev, curr):
    gray1 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    return diff.sum() < MOTION_THRESHOLD


def run_traffic_signal(priority_road, results):
    print("\n🚦 TRAFFIC SIGNAL STATUS:")

    # ---------------- GREEN PHASE ----------------
    for road in results:
        if road == priority_road:
            print(f"{road}: 🟢 GREEN")
        else:
            print(f"{road}: 🔴 RED")

    MIN_GREEN = 10
    MAX_GREEN = 60
    
    raw_time = int(results[priority_road]["score"] * 2)
    
    green_time = max(MIN_GREEN, min(raw_time, MAX_GREEN))

    print(f"\n⏳ GREEN Time for {priority_road}: {green_time} seconds")
    start = time.time()

    while time.time() - start < green_time:
        cv2.waitKey(1)

    # ---------------- YELLOW PHASE ----------------#
    
    YELLOW_TIME = 3
    
    print(f"\n⚠️ {priority_road}: 🟡 YELLOW ({YELLOW_TIME} seconds)")
    
    start = time.time()
    
    while time.time() - start < YELLOW_TIME:
        cv2.waitKey(1)

    # ---------------- ALL RED (OPTIONAL SAFETY) ----------------
    ALL_RED_TIME = 1
    print("\n🚦 ALL RED (1 second safety buffer)")

    start = time.time()

    while time.time() - start < ALL_RED_TIME:
        cv2.waitKey(1)


# Initialize
for road in caps:
    prev_frames[road] = None
    last_capture_time[road] = 0
    cycle_data[road] = []
    was_waiting[road] = False


while True:

    for road, cap in caps.items():
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if lanes[road] is None:

            lanes[road] = detect_lanes(frame, road)

        traffic_stopped = False

        # Detect traffic condition
        if prev_frames[road] is not None:
            if is_traffic_stopped(prev_frames[road], frame):
                traffic_stopped = True

        prev_frames[road] = frame.copy()

        current_time = time.time()

        # ----------------------------
        # Collect data during congestion
        # ----------------------------
        if traffic_stopped:
            was_waiting[road] = True

            if current_time - last_capture_time[road] > 5:
                last_capture_time[road] = current_time

                results = model(frame, conf=0.25)[0]

                counts = {"car": 0, "bike": 0, "bus": 0, "truck": 0}

                if results.boxes is not None:
                    for box in results.boxes:
                        cls = int(box.cls[0])

                        if cls in vehicle_map:
                            counts[vehicle_map[cls]] += 1

                cycle_data[road].append(counts)

        # ----------------------------
        # End of cycle → store result
        # ----------------------------
        else:
            if was_waiting[road] and cycle_data[road]:

                max_counts = {
                    "car": max(d["car"] for d in cycle_data[road]),
                    "bike": max(d["bike"] for d in cycle_data[road]),
                    "bus": max(d["bus"] for d in cycle_data[road]),
                    "truck": max(d["truck"] for d in cycle_data[road]),
                }

                completed_roads[road] = {
                    "counts": max_counts,
                    "lanes": lanes[road]
                }

                print(f"{road} Data:", max_counts)

                # Reset per-road cycle
                cycle_data[road].clear()
                was_waiting[road] = False

    # ----------------------------
    # FINAL DECISION (ALL ROADS READY)
    # ----------------------------
    if len(completed_roads) == 4:

        results = calculate_congestion(completed_roads)

        print("\n🚦 Congestion Scores:")
        for r, val in results.items():
            print(f"{r}: {val['score']}")

        priority = get_priority_road(results)

        run_traffic_signal(priority, results)

        # Reset for next full cycle
        completed_roads.clear()

    # Exit
    if cv2.waitKey(1) & 0xFF == 27:
        break


# Release all cameras
for cap in caps.values():
    cap.release()

cv2.destroyAllWindows()