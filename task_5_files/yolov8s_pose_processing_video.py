from multiprocessing import Pool, cpu_count, set_start_method
from ultralytics import YOLO
import numpy as np
import argparse
import time
import uuid
import cv2
import os

model = None

class VideoResourceManager:
    """RAII class for managing video capture and writer resources."""
    def __init__(self, input_path = None, output_path = None, width = None, height = None, fps = None):
        self.cap = None
        self.writer = None
        if input_path:
            self.cap = cv2.VideoCapture(input_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video: {input_path}")
        if output_path and width and height and fps:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to create output video: {output_path}")

    def __del__(self):
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()

def initialize_model(model_path):
    """Model initialization for each process."""
    global model
    model = YOLO(model_path)

def process_frame(frame):
    """Process a single frame with YOLOv8s-pose (модель уже загружена)."""
    global model
    results = model(frame, verbose = False)
    annotated_frame = results[0].plot()  # Draw keypoints
    return annotated_frame

def single_thread_processing(video_path, output_path, model_path):
    """Process video in single-thread mode."""
    manager = VideoResourceManager(video_path, output_path, 640, 480, 30)
    cap = manager.cap
    writer = manager.writer
    model = YOLO(model_path)
    
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose = False)
        annotated_frame = results[0].plot()
        writer.write(annotated_frame)
    
    end_time = time.time()
    processing_time = end_time - start_time
    return processing_time

def multi_thread_processing(video_path, output_path, model_path, num_processes):
    """Process video in multi-process mode."""
    manager = VideoResourceManager(video_path, output_path, 640, 480, 30)
    cap = manager.cap
    frames = []
    
    # Read all frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    start_time = time.time()
    with Pool(processes=num_processes, initializer = initialize_model, initargs = (model_path,)) as pool:
        annotated_frames = pool.map(process_frame, frames)
    
    # Write processed frames
    for frame in annotated_frames:
        manager.writer.write(frame)
    
    end_time = time.time()
    processing_time = end_time - start_time
    return processing_time

def main():
    parser = argparse.ArgumentParser(description = "YOLOv8s-pose video processing")
    parser.add_argument('--video', type = str, required = True, help = 'Path to input video')
    parser.add_argument('--mode', choices = ['single', 'multi'], required = True, help = 'Processing mode')
    parser.add_argument('--output', type = str, required = True, help = 'Path to output video')
    args = parser.parse_args()

    model_path = 'yolov8s-pose.pt'
    
    if args.mode == 'single':
        print("Running in single-thread mode...")
        processing_time = single_thread_processing(args.video, args.output, model_path)
        print(f"Processing time: {processing_time:.2f} seconds")
    else:
        print("Running in multi-process mode...")
        # Test different numbers of processes to find optimal
        optimal_time = float('inf')
        optimal_processes = 1
        max_processes = min(cpu_count(), 4)  # Limit to 8 (4) processes or CPU count
        process_times = {}  # Store times for each number of processes
        
        for num_processes in range(1, max_processes + 1):
            print(f"Testing with {num_processes} processes...")
            temp_output = f"temp_{uuid.uuid4()}.mp4"
            processing_time = multi_thread_processing(args.video, temp_output, model_path, num_processes)
            process_times[num_processes] = processing_time
            print(f"Time with {num_processes} processes: {processing_time:.2f} seconds")
            if processing_time < optimal_time:
                optimal_time = processing_time
                optimal_processes = num_processes
            if os.path.exists(temp_output):
                os.remove(temp_output)
        
        # Print summary of all process times
        print("\nSummary of processing times:")
        for num_processes, time_taken in process_times.items():
            print(f"{num_processes} processes: {time_taken:.2f} seconds")
        
        # Run with optimal number of processes
        print(f"\nOptimal number of processes: {optimal_processes}")
        processing_time = multi_thread_processing(args.video, args.output, model_path, optimal_processes)
        print(f"Final processing time with {optimal_processes} processes: {processing_time:.2f} seconds")

if __name__ == '__main__':
    set_start_method('spawn', force = True)
    main()