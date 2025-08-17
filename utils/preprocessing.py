import cv2
from datetime import datetime

points = []

def select_lines(event, x, y, flags, param):
    """Mouse click callback for selecting line points."""
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"[INFO] Point selected: {(x, y)}")

def draw_lines(frame, points):
    """Draw incoming and outgoing lines if selected."""
    if len(points) >= 2:
        cv2.line(frame, points[0], points[1], (0, 255, 0), 3)
        cv2.putText(frame, "INCOMING", (points[0][0], points[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if len(points) >= 4:
        cv2.line(frame, points[2], points[3], (0, 0, 255), 3)
        cv2.putText(frame, "OUTGOING", (points[2][0], points[2][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def slow_video_playback(cap, window_name, delay=50):
    """Plays video slower for selecting points."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        temp_frame = frame.copy()
        draw_lines(temp_frame, points)
        cv2.imshow(window_name, temp_frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('s') and len(points) >= 4:
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

def current_timestamp():
    """Returns current timestamp as string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
