import os
import cv2
import torch
import json
import pandas as pd
import gpxpy
import gpxpy.gpx
from exif import Image
from ultralytics import YOLO
import numpy as np

class pTrack:
    def __init__(self, directory):
        if not os.path.isdir(directory):
            raise ValueError(f"Invalid directory: {directory}")
        
        self.directory = directory
        self.model = YOLO("yolov8n.pt")  # Load YOLOv8 model

    def load(self, file_path):
        if not os.path.isfile(file_path):
            raise ValueError(f"Invalid file path: {file_path}")
        
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            return self.process_image(file_path)
        elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
            return self.process_video(file_path)
        else:
            raise ValueError("Unsupported file type")

    def process_image(self, image_path):
        try:
            with open(image_path, 'rb') as img_file:
                img = Image(img_file)
                
            if img.has_exif:
                lat, lon = self.extract_gps(img)
                
                results = self.model(image_path, size=1280)
                detections = results[0].boxes.data.cpu().numpy()
                classes = detections[:, 5] if detections.shape[0] > 0 else []
                plastic_count = np.sum(classes == 0)
                
                data = {
                    "filename": os.path.basename(image_path),
                    "latitude": lat,
                    "longitude": lon,
                    "plastic_count": int(plastic_count)
                }
                
                self.export_geojson([data], "output.geojson")
                return data
            else:
                raise ValueError("No EXIF data found")
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        results_data = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 5 == 0:  # Process every 5th frame
                results = self.model(frame)
                detections = results[0].boxes.data.cpu().numpy()
                classes = detections[:, 5] if detections.shape[0] > 0 else []
                plastic_count = np.sum(classes == 0)
                
                results_data.append({"frame": frame_count, "plastic_count": int(plastic_count)})
        
        cap.release()
        self.export_json(results_data, "video.json")
        return results_data

    def extract_gps(self, img):
        try:
            lat = img.gps_latitude[0] + img.gps_latitude[1] / 60 + img.gps_latitude[2] / 3600
            lon = img.gps_longitude[0] + img.gps_longitude[1] / 60 + img.gps_longitude[2] / 3600
            return round(lat, 6), round(lon, 6)
        except AttributeError:
            raise ValueError("Missing GPS coordinates in EXIF data")

    def export_geojson(self, data, filename):
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [entry["longitude"], entry["latitude"]]
                    },
                    "properties": {
                        "filename": entry["filename"],
                        "plastic_count": entry["plastic_count"]
                    }
                } for entry in data
            ]
        }
        
        with open(filename, "w") as f:
            json.dump(geojson_data, f, indent=4)
        print(f"GeoJSON saved: {filename}")
    
    def export_json(self, data, filename):
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"JSON saved: {filename}")


# Usage example
directory = "C:\\Users\\Admin\\Documents\\GitHub\\pLitter\\plitter\\input"
ptracker = pTrack(directory)

image_path = "C:\\Users\\Admin\\Documents\\GitHub\\pLitter\\plitter\\input\\2_4CCPW.jpg"
video_path = "C:\\Users\\Admin\\Documents\\GitHub\\pLitter\\plitter\\input\\video.mp4"


image_result = ptracker.load(image_path)
print("Image Result:", image_result)

video_result = ptracker.load(video_path)
print("Video Processing Done")
