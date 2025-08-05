import cv2

def resize_frame(frame, width=1280):
    return cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])))
