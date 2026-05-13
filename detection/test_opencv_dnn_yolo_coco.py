import cv2
import numpy as np

# Load the YOLOv3 model with OpenCV
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

# Load the COCO class labels
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Load an image for object detection
image = cv2.imread("kite.jpg")

# Get the height and width of the image
height, width = image.shape[:2]

# Create a blob from the image (preprocess)
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

# Set the input to the neural network
net.setInput(blob)

# Get the output layer names
layer_names = net.getUnconnectedOutLayersNames()

# Run forward pass through the network
outs = net.forward(layer_names)

# Initialize lists to store detected objects' class IDs, confidences, and bounding boxes
class_ids = []
confidences = []
boxes = []

# Minimum confidence threshold for object detection
conf_threshold = 0.2

# Iterate through each output layer
for out in outs:
    # Iterate through each detection in the output
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold:
            # Scale the bounding box coordinates to the original image size
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Calculate the top-left corner of the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

# Non-maximum suppression to remove redundant boxes
nms_threshold = 0.4
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Draw bounding boxes and labels on the image
for i in indices:
    #i = i[0]
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = confidences[i]
    color = (0, 255, 0)  # Green color for the bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the resulting image
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

