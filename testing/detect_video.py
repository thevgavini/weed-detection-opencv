import cv2
import os
import numpy as np

# Define file paths
labelsPath = 'obj.names'
weightsPath = 'crop_weed_detection.weights'
configPath = 'crop_weed.cfg'
videoPath = 'crop_weed_video.mp4'  # Change this if using a different video file

# Check if required files exist
for file in [labelsPath, weightsPath, configPath, videoPath]:
    if not os.path.exists(file):
        print(f"[ERROR]    : Missing file -> {file}")
        exit(1)

# Load class labels
LABELS = open(labelsPath).read().strip().split("\n")

# Generate random colors for bounding boxes
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Load YOLO model
print("[INFO]     : Loading YOLO model...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Set confidence & threshold parameters
confi = 0.5  # Confidence threshold
thresh = 0.5  # Non-Maximum Suppression (NMS) threshold

# Get output layer names (Fix applied)
ln = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()

# Fix for OpenCV 4.11+
if len(unconnected_layers.shape) == 1:  # 1D array case
    ln = [ln[i - 1] for i in unconnected_layers]
else:  # 2D array case
    ln = [ln[i[0] - 1] for i in unconnected_layers]

# Use webcam instead of video (Uncomment if needed)
# cap = cv2.VideoCapture(0)

# Load video file
cap = cv2.VideoCapture(videoPath)

# Loop through video frames
while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break  # End loop if video is done

    (H, W) = image.shape[:2]

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (512, 512), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    # Process YOLO detections
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confi:
                # Scale bounding box coordinates
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply Non-Maximum Suppression (NMS)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi, thresh)

    # Draw bounding boxes if detections exist
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            print(f"[OUTPUT]   : Detected -> {LABELS[classIDs[i]]}")
            text = f"{LABELS[classIDs[i]]}: {confidences[i]:.4f}"
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display the video frame
    cv2.imshow('YOLO Weed Detection', image)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("[STATUS]   : Video processing completed.")
