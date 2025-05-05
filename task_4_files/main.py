from typing import Tuple, Optional
import numpy as np
import threading
import argparse
import logging
import queue
import time
import cv2
import os

log_path = "log"
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_path, 'errors_logs.log'),
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Sensor:
    def get(self):
        raise NotImplementedError("Subclass must implement method get()")

class SensorX(Sensor):
    '''Sensor X'''
    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0
    
    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data

class SensorCam(Sensor):
    def __init__(self, camera_name: str, resolution: Tuple[int, int]):
        try:
            self.cap = cv2.VideoCapture(camera_name)
            if not self.cap.isOpened():
                raise RuntimeError(f"Can't open camera {camera_name}")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        except Exception as e:
            logging.error(f"Camera init error: {str(e)}")
            raise
    
    def get(self) -> Optional[np.ndarray]:
        try:
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("Failed to grab frame from camera")
                return None
            return frame
        except Exception as e:
            logging.error(f"Camera read error: {str(e)}")
            return None
    
    def __del__(self):
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

class WindowImage:
    def __init__(self, freq: float):
        self.freq = freq
        self.last_show = 0
        self.frame = None
        self.sensors_data = [0, 0, 0]
        try:
            cv2.namedWindow("Sensor Display", cv2.WINDOW_NORMAL)
        except Exception as e:
            logging.error(f"Window init error: {str(e)}")
            raise
    
    def update(self, frame, *sensor_values):
        try:
            if frame is not None:
                self.frame = frame.copy()
            
            for i, val in enumerate(sensor_values):
                if val is not None:
                    self.sensors_data[i] = val
            
            if time.time() - self.last_show >= 1/self.freq:
                if self.frame is not None:
                    display_frame = self.frame.copy()
                    for i, val in enumerate(self.sensors_data):
                        cv2.putText(
                            display_frame, 
                            f"Sensor{i}: {val}", 
                            (10, 30 + i*40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, 
                            (255, 255, 255), 
                            2
                        )
                    cv2.imshow("Sensor Display", display_frame)
                    self.last_show = time.time()
        except Exception as e:
            logging.error(f"WindowImage update error: {str(e)}")

def worker(sensor: Sensor, q: queue.Queue):
    while True:
        try:
            data = sensor.get()
            q.put(data)
        except Exception as e:
            logging.error(f"Sensor worker error: {str(e)}")
            break

def get_last(q: queue.Queue):
    latest = None
    try:
        while not q.empty():
            latest = q.get_nowait()
    except queue.Empty:
        pass
    return latest

def main():
    parser = argparse.ArgumentParser(description="Sensor Display System")
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--resolution", default="640x480")
    parser.add_argument("--frequency", type=float, default=30.0)
    args = parser.parse_args()
    
    try:
        width, height = map(int, args.resolution.split('x'))
        
        devices = [
            SensorX(0.01),
            SensorX(0.1),
            SensorX(1.0),
            SensorCam(args.camera, (width, height))
        ]
        
        queues = [queue.Queue() for _ in devices]
        
        threads = []
        for device, q in zip(devices, queues):
            thread = threading.Thread(
                target=worker,
                args=(device, q),
                daemon=True
            )
            thread.start()
            threads.append(thread)
        
        window = WindowImage(args.frequency)
        
        while True:
            frame = get_last(queues[3])
            sensor_values = [get_last(q) for q in queues[:3]]
            
            window.update(frame, *sensor_values)
            
            key = cv2.waitKey(1)
            if key != -1:
                if key == ord('q'):
                    break
                logging.warning(
                    f"Invalid exit key: {chr(key) if 32 <= key <= 126 else 'non-printable'} (code: {key})"
                )
                
    except Exception as e:
        logging.critical(f"Critical error: {str(e)}", exc_info=True)
        raise
    finally:
        cv2.destroyAllWindows()
        logging.info("Application shutdown")

if __name__ == "__main__":
    main()

'''
Example usage: [python main.py --camera /dev/video0 --resolution 640x480 --frequency 60]
'''