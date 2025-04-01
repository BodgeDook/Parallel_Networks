import argparse
import logging
import os
import queue
import threading
import time
import cv2
import numpy as np
from abc import ABC

log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(
    filename=os.path.join(log_dir, "sensor_display.log"),
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class Sensor(ABC):
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")

class SensorX(Sensor):
    def __init__(self, delay: float):
        self.delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self.delay)
        self._data += 1
        return self._data

class SensorCam(Sensor):
    def __init__(self, camera_name: str, resolution: tuple):
        self.camera_name = camera_name
        self.resolution = resolution
        self.cap = None
        try:
            self.cap = cv2.VideoCapture(self.camera_name)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_name}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        except Exception as e:
            logging.error(f"Camera initialization error {self.camera_name}: {str(e)}")
            if self.cap:
                self.cap.release()
            raise

    def get(self):
        try:
            if not self.cap or not self.cap.isOpened():
                logging.error("Camera not initialized or has failed")
                return None
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to read frame from camera")
                return None
            return frame
        except Exception as e:
            logging.error(f"Camera read error: {str(e)}")
            return None

    def __del__(self):
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception as e:
            logging.error(f"Error releasing camera: {str(e)}")

class WindowImage:
    def __init__(self, display_freq: float):
        self.display_freq = display_freq
        self.window_name = "Sensor Display"
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        except Exception as e:
            logging.error(f"Failed to create window: {str(e)}")
            raise

    def show(self, img):
        try:
            cv2.imshow(self.window_name, img)
            return True
        except Exception as e:
            logging.error(f"Display error: {str(e)}")
            return False

    def __del__(self):
        try:
            cv2.destroyWindow(self.window_name)
        except Exception as e:
            logging.error(f"Failed to close window: {str(e)}")

def sensor_reader(sensor, data_queue):
    while not stop_event.is_set():
        try:
            data = sensor.get()
            if data is not None:
                try:
                    data_queue.put(data, block=False)
                except queue.Full:
                    try:
                        data_queue.get_nowait()
                    except queue.Empty:
                        pass
                    data_queue.put(data, block=False)
        except Exception as e:
            logging.error(f"Sensor read error: {str(e)}")
            stop_event.set()
            break

def parse_args():
    parser = argparse.ArgumentParser(description="Sensor Display Program")
    parser.add_argument("--camera", default="/dev/video0", help="Camera device name (e.g., /dev/video0)")
    parser.add_argument("--resolution", default="1280x720", help="Camera resolution (e.g., 1280x720)")
    parser.add_argument("--display-freq", type=float, default=30.0, help="Display frequency in Hz")
    args = parser.parse_args()
    
    try:
        width, height = map(int, args.resolution.split("x"))
        args.resolution = (width, height)
    except ValueError:
        logging.error("Invalid resolution format. Use WIDTHxHEIGHT (e.g., 1280x720)")
        raise ValueError("Invalid resolution format")
    
    return args

def main():
    global stop_event
    stop_event = threading.Event()

    try:
        args = parse_args()

        sensor0 = SensorX(0.01)
        sensor1 = SensorX(0.1)
        sensor2 = SensorX(1.0)
        camera = None
        try:
            camera = SensorCam(args.camera, args.resolution)
        except Exception as e:
            print(f"Camera initialization error: {str(e)}")
            return

        queues = {
            "camera": queue.Queue(maxsize=1),
            "sensor0": queue.Queue(maxsize=1),
            "sensor1": queue.Queue(maxsize=1),
            "sensor2": queue.Queue(maxsize=1)
        }

        threads = []
        threads.append(threading.Thread(target=sensor_reader, args=(camera, queues["camera"])))
        threads.append(threading.Thread(target=sensor_reader, args=(sensor0, queues["sensor0"])))
        threads.append(threading.Thread(target=sensor_reader, args=(sensor1, queues["sensor1"])))
        threads.append(threading.Thread(target=sensor_reader, args=(sensor2, queues["sensor2"])))
        for t in threads:
            t.daemon = True
            t.start()

        window = None
        try:
            window = WindowImage(args.display_freq)
        except Exception as e:
            print(f"Window initialization error: {str(e)}")
            stop_event.set()
            return

        latest_values = np.zeros(3, dtype=np.int32)
        sensor_names = ["sensor0", "sensor1", "sensor2"]
        text_positions = np.array([[10, 30], [10, 60], [10, 90]], dtype=np.int32)
        latest_frame = None

        display_interval = 1.0 / args.display_freq
        last_display_time = time.time()

        while not stop_event.is_set():
            current_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            try:
                latest_frame = queues["camera"].get_nowait()
            except queue.Empty:
                pass

            for i, sensor_name in enumerate(sensor_names):
                try:
                    latest_values[i] = queues[sensor_name].get_nowait()
                except queue.Empty:
                    pass

            if current_time - last_display_time >= display_interval and latest_frame is not None:
                frame = latest_frame.copy()
                for i in range(len(sensor_names)):
                    cv2.putText(frame, f"{sensor_names[i]}: {latest_values[i]}", 
                                (text_positions[i, 0], text_positions[i, 1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if not window.show(frame):
                    stop_event.set()
                    break
                last_display_time = current_time

            # time.sleep(0.001) # Small sleep to prevent high CPU usage

    except KeyboardInterrupt:
        print("Program interrupted by user (Ctrl+C)")
        logging.info("Program interrupted by user (Ctrl+C)")
        stop_event.set()

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        logging.error(f"Unexpected error: {str(e)}")
        stop_event.set()

    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=1.0)
        cv2.destroyAllWindows()
        if 'window' in locals():
            del window
        if 'camera' in locals():
            del camera

if __name__ == "__main__":
    main()