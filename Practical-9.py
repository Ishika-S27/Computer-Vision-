import cv2
import imutils
from fer.fer import FER
from google.colab.patches import cv2_imshow
from google.colab import files
uploaded = files.upload()
image_file = list(uploaded.keys())[0]
image = cv2.imread(image_file)
image = imutils.resize(image, width=400)
cv2_imshow(image)
# Initialize the emotion detector. The FER library
internally uses a CNN trained on FER-2013 dataset.
detector = FER(mtcnn=True)
#Detect faces and emotions.
results = detector.detect_emotions(image)
for result in results:
(x, y, w, h) = result[&quot;box&quot;]
emotions = result[&quot;emotions&quot;]
emotion = max(emotions, key=emotions.get)
score = emotions[emotion]
cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0),
2)
text = f&quot;{emotion}: {score:.2f}&quot;
cv2.putText(image, text, (x, y-10),
cv2.FONT_HERSHEY_SIMPLEX, 0.6,

(0,255,0), 2)
cv2_imshow(image)
