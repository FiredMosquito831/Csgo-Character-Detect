import cv2
import numpy as np
import mss
from PyQt5 import QtCore, QtGui, QtWidgets
import torch
import keyboard
import time
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO

# Define constants for mouse input simulation
confidence = 0.45
frame_skip = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

# Load your trained YOLO model
model = YOLO('../GtaModels/S45images/train4/weights/best.engine') # current best

class DetectionThread(QThread):
    update_overlay_signal = pyqtSignal(np.ndarray)

    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor
        self.is_paused = False
        self.frame_count = frame_skip  # Counter for skipping frames

    def toggle_pause(self):
        """Toggle the pause state of the thread."""
        self.is_paused = not self.is_paused
        print("Program Paused" if self.is_paused else "Program Resumed")

    def run(self):
        sct = mss.mss()
        while True:
            if self.is_paused:
                continue

            self.frame_count += 1
            if self.frame_count % 2 != 0:  # Skip every second frame
                continue

            # Capture screen
            screen = np.array(sct.grab(self.monitor))
            screen_bgr = cv2.cvtColor(screen, cv2.COLOR_RGBA2BGR)
            screen_bgr_resized = cv2.resize(screen_bgr, (640, 480))  # Reduce resolution for speed

            # Perform YOLO detection
            results = model(screen_bgr_resized, conf=confidence)  # Set confidence threshold
            overlay_img = self.process_detections(results)

            # Emit the updated overlay image
            self.update_overlay_signal.emit(overlay_img)

    def process_detections(self, results):
        """Process YOLO results and create overlay image."""
        overlay_img = np.zeros((480, 640, 3), dtype=np.uint8)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_score = box.conf.item()  # Confidence score

                # Draw rectangle on overlay
                cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (125, 255, 35), 2)
                label = f'{conf_score:.2f}'
                cv2.putText(overlay_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return overlay_img

class ScreenCaptureOverlay(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # Set window properties to make it transparent and non-clickable
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint |
                            QtCore.Qt.WindowStaysOnTopHint |
                            QtCore.Qt.WindowTransparentForInput)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)

        # Use MSS for capturing screen
        self.monitor = mss.mss().monitors[1]  # Use primary monitor

        # Set window size based on monitor
        self.setGeometry(self.monitor["left"], self.monitor["top"], self.monitor["width"], self.monitor["height"])

        # Initialize the pixmap for the overlay
        self.pixmap = QtGui.QPixmap(self.size())
        self.pixmap.fill(QtCore.Qt.transparent)

        # Initialize detection thread
        self.detection_thread = DetectionThread(self.monitor)
        self.detection_thread.update_overlay_signal.connect(self.update_overlay_with_detections)
        self.detection_thread.start()

        # Set up global key listener
        keyboard.add_hotkey(']', self.detection_thread.toggle_pause)  # Register the toggle function

    def update_overlay_with_detections(self, overlay_img):
        try:
            # Convert the overlay image to QImage
            h, w, ch = overlay_img.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(overlay_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

            # Create a QPixmap from QImage and set it as the overlay
            self.pixmap = QtGui.QPixmap.fromImage(qt_image)
            self.update()  # Trigger a repaint
        except Exception as e:
            print(f"Error in update_overlay_with_detections: {e}")

    def paintEvent(self, event):
        try:
            painter = QtGui.QPainter(self)
            painter.setOpacity(0.4)  # Set transparency for the overlay
            painter.drawPixmap(self.rect(), self.pixmap)

            # Draw the reticle (red dot) at the center of the screen
            center_x = self.width() // 2
            center_y = self.height() // 2
            reticle_radius = 2
            painter.setBrush(QtGui.QBrush(QtCore.Qt.red))
            painter.setPen(QtGui.QPen(QtCore.Qt.red))
            painter.drawEllipse(center_x - reticle_radius, center_y - reticle_radius, reticle_radius * 2, reticle_radius * 2)
        except Exception as e:
            print(f"Error in paintEvent: {e}")

# PyQt5 Application
app = QtWidgets.QApplication([])

# Create and show the overlay window
overlay_window = ScreenCaptureOverlay()
overlay_window.show()

# Start the application event loop
app.exec_()
