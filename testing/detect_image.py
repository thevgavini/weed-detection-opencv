from PIL import Image  # Explicitly import Image module
import cv2
import os
import numpy as np
from google.colab.patches import cv2_imshow
import time

# Define paths
labelsPath = 'obj.names'
weightsPath = 'crop_weed_detection.weights'
configPath = 'crop_weed.cfg'
imagePath = 'images/weed_2.jpeg'

# Check if required files exist
for file in [labelsPath, weightsPath, configPath, imagePath]:
    if not os.path.exists(file):
        print(f"[ERROR]    : Missing file -> {file}")
        exit(1)

# Load class labels
LABELS = open(labelsPath).read().strip().split("\n")

# Load YOLO model
print("[INFO]     : Loading YOLO model...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Load input image
image = cv2.imread(imagePath)
(H, W) = image.shape[:2]

# Parameters
confi = 0.5  # Confidence threshold
thresh = 0.5  # Non-Maximum Suppression (NMS) threshold

# Get output layer names
ln = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()

# Fix for OpenCV 4.11+
if len(unconnected_layers.shape) == 1:
    ln = [ln[i - 1] for i in unconnected_layers]
else:
    ln = [ln[i[0] - 1] for i in unconnected_layers]

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (512, 512), swapRB=True, crop=False)
net.setInput(blob)

# Run YOLO detection
start_time = time.time()
layerOutputs = net.forward(ln)
end_time = time.time()

print(f"[INFO]     : YOLO took {end_time - start_time:.6f} seconds")

# Initialize lists for detections
boxes = []
confidences = []
classIDs = []

# Process detections
for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > confi:
            # Extract bounding box coordinates
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # Store correct bounding box format
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# Apply Non-Maximum Suppression (NMS)
if len(boxes) > 0:
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi, thresh)
else:
    idxs = []

# Display detailed detection info
if len(idxs) > 0:
    print("\n[DETECTIONS] :")
    for i in idxs.flatten():
        detected_label = LABELS[classIDs[i]]
        print(f"- {detected_label} (Confidence: {confidences[i]:.2f})")

# Determine the final verdict (most frequent class detected)
if len(classIDs) > 0:
    most_frequent_class = max(set(classIDs), key=classIDs.count)
    detected_label = LABELS[most_frequent_class]  # Get the final class
    print(f"\n{detected_label}")  # Print only "Weed" or "Crop"
else:
    print("\nUnknown")  # If nothing is detected

print("\n[STATUS]   : Completed")


