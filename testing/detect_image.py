import cv2
import numpy as np
import time
import os

# Define paths
labelsPath = 'obj.names'
weightsPath = 'crop_weed_detection.weights'
configPath = 'crop_weed.cfg'
imagePath = 'images/weed_1.jpeg'

# Check if required files exist
for file in [labelsPath, weightsPath, configPath, imagePath]:
    if not os.path.exists(file):
        print(f"[ERROR]    : Missing file -> {file}")
        exit(1)

# Load class labels
LABELS = open(labelsPath).read().strip().split("\n")

# Color selection for bounding boxes
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Load YOLO model
print("[INFO]     : Loading YOLO model...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Load input image
image = cv2.imread(imagePath)
(H, W) = image.shape[:2]

# Parameters
confi = 0.5  # Confidence threshold
thresh = 0.5  # Non-Maximum Suppression (NMS) threshold

# Get output layer names (Fix applied)
ln = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()

# Fix for OpenCV 4.11.0+
if len(unconnected_layers.shape) == 1:  # 1D array case
    ln = [ln[i - 1] for i in unconnected_layers]
else:  # 2D array case
    ln = [ln[i[0] - 1] for i in unconnected_layers]

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (512, 512), swapRB=True, crop=False)
net.setInput(blob)

# Run YOLO detection
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()
print(f"[INFO]     : YOLO took {end - start:.6f} seconds")

# Initialize bounding boxes, confidences, and class IDs
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
            # Scale bounding box coordinates
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")

            # Get top-left coordinates
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # Save detection details
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# Apply Non-Maximum Suppression (NMS)
idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi, thresh)
print("[INFO]     : Detections done, drawing bounding boxes...")

# Draw bounding boxes if detections exist
if len(idxs) > 0:
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        print(f"[OUTPUT]   : Detected label -> {LABELS[classIDs[i]]}")
        print(f"[ACCURACY] : {confidences[i]:.4f}")

        text = f"{LABELS[classIDs[i]]}: {confidences[i]:.4f}"
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# Show the output image
cv2.imshow('Output', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("[STATUS]   : Completed")
print("[END]")
