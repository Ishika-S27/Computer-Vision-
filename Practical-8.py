import cv2
import numpy as np
# Load YOLO model (pre-trained weights and config)
net = cv2.dnn.readNet(&quot;yolov3.weights&quot;, &quot;yolov3.cfg&quot;)
# Load class names
with open(&quot;coco.names&quot;, &quot;r&quot;) as f:
classes = [line.strip() for line in f.readlines()]
# Read input image
image = cv2.imread(&quot;input.jpg&quot;)
height, width, channels = image.shape
# Preprocess image
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416),
swapRB=True, crop=False)
net.setInput(blob)
# Get output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in
net.getUnconnectedOutLayers()]
# Forward pass
outputs = net.forward(output_layers)
# Process detections
for output in outputs:
for detection in output:
scores = detection[5:]
class_id = np.argmax(scores)
confidence = scores[class_id]
if confidence &gt; 0.5:
center_x = int(detection[0] * width)
center_y = int(detection[1] * height)
w = int(detection[2] * width)
h = int(detection[3] * height)
x = int(center_x - w / 2)
y = int(center_y - h / 2)
label = str(classes[class_id])
cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
cv2.putText(image, label, (x, y-10),
cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
# Show output
cv2.imshow(&quot;Object Recognition&quot;, image)
cv2.waitKey(0)
cv2.destroyAllWindows()
