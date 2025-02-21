from spatial import pTrack
import torch

# Path to the directory containing images/videos
data_directory = "input"

# Load model (Example: YOLOv5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize the object
tracker = pTrack(data_directory)

# Load images and videos
tracker.load()

# Run detection
tracker.detect(model)

# Export output
tracker.export("output.json", with_predictions=True)

print("Detection complete. Results saved to output.json")
