import cv2
from ultralytics import YOLO
import os
from collections import defaultdict
from utils import select_lines, slow_video_playback, points
from logs import init_log, log_event

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


model = YOLO('yolo11l.pt')
class_list = model.names

video_source = input("Enter video file path or RTSP stream link: ")
cap = cv2.VideoCapture(video_source)

init_log()

incoming_count = 0
outgoing_count = 0
incoming_type_count = defaultdict(int)
outgoing_type_count = defaultdict(int)
incoming_crossed_ids = set()
outgoing_crossed_ids = set()
last_positions = {}

cv2.namedWindow("Dual-Lane Vehicle Counting")
cv2.setMouseCallback("Dual-Lane Vehicle Counting", select_lines)

print("[INFO] Click 2 points for INCOMING line (green), then 2 points for OUTGOING line (red). Press 's' to start detection.")

slow_video_playback(cap, "Dual-Lane Vehicle Counting", delay=80)

incoming_line_start, incoming_line_end = points[0], points[1]
outgoing_line_start, outgoing_line_end = points[2], points[3]

cap.release()
cap = cv2.VideoCapture(video_source)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.line(frame, incoming_line_start, incoming_line_end, (0, 255, 0), 3)
    cv2.putText(frame, "INCOMING", (incoming_line_start[0], incoming_line_start[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.line(frame, outgoing_line_start, outgoing_line_end, (0, 0, 255), 3)
    cv2.putText(frame, "OUTGOING", (outgoing_line_start[0], outgoing_line_start[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    results = model.track(frame, persist=True, classes=[1, 2, 3, 5, 7])

    if results[0].boxes.data is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        current_positions = {}

        for box, track_id, class_idx in zip(boxes, track_ids, class_indices):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            current_positions[track_id] = (cx, cy)

            class_name = class_list[class_idx]
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if track_id in last_positions:
                prev_x, prev_y = last_positions[track_id]
                direction = "down" if cy > prev_y else "up"

                if incoming_line_start[0] < cx < incoming_line_end[0]:
                    if direction == "down" and cy > incoming_line_start[1] and track_id not in incoming_crossed_ids:
                        incoming_crossed_ids.add(track_id)
                        incoming_count += 1
                        incoming_type_count[class_name] += 1
                        log_event("Incoming", class_name, incoming_count)

                if outgoing_line_start[0] < cx < outgoing_line_end[0]:
                    if direction == "up" and cy < outgoing_line_start[1] and track_id not in outgoing_crossed_ids:
                        outgoing_crossed_ids.add(track_id)
                        outgoing_count += 1
                        outgoing_type_count[class_name] += 1
                        log_event("Outgoing", class_name, outgoing_count)

        last_positions = current_positions

    y_offset = 50
    cv2.putText(frame, f"INCOMING: {incoming_count}", (50, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    for vtype, count in incoming_type_count.items():
        y_offset += 30
        cv2.putText(frame, f"  {vtype}: {count}", (50, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_offset += 40
    cv2.putText(frame, f"OUTGOING: {outgoing_count}", (50, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    for vtype, count in outgoing_type_count.items():
        y_offset += 30
        cv2.putText(frame, f"  {vtype}: {count}", (50, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    y_offset += 40
    cv2.putText(frame, f"GRAND TOTAL: {incoming_count + outgoing_count}", (50, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Dual-Lane Vehicle Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
