# Required imports 
from collections import deque
import numpy as np
import cv2
import pyttsx3
import time

# Parameters class include important paths and constants
class Parameters:
    def _init_(self):
        self.CLASSES = open("model/action_recognition_kinetics.txt"
                            ).read().strip().split("\n")
        self.ACTION_RESNET = 'model/resnet-34_kinetics.onnx'
        #self.VIDEO_PATH = None
        self.VIDEO_PATH = "test/killing.mp4"
        # SAMPLE_DURATION is maximum deque size
        self.SAMPLE_DURATION = 16
        self.SAMPLE_SIZE = 112
        self.COOLDOWN_TIME = 5  # Cooldown time in seconds

# Initialise instance of Class Parameter``
param = Parameters() 

# A Double ended queue to store our frames captured and with time
# old frames will pop
# out of the deque
captures = deque(maxlen=param.SAMPLE_DURATION)

# load the human activity recognition model
print("[INFO] loading human activity recognition model...")
net = cv2.dnn.readNet(model=param.ACTION_RESNET)

print("[INFO] accessing video stream...")
# Take video file as input if given else turn on web-cam
# So, the input should be mp4 file or live web-cam video
vs = cv2.VideoCapture(param.VIDEO_PATH if param.VIDEO_PATH else 0)

# Initialize the pyttsx3 engine for speech output
engine = pyttsx3.init()

# Check available voices
print("[INFO] Available voices:", engine.getProperty('voices'))

# Set a different voice if available
# engine.setProperty('voice', 'english-us')

# Flag to check if pushup action was detected
pushup_detected = False

# Timestamp for last alert
last_alert_time = 0

while True:
    # Loop over and read capture from the given video input
    (grabbed, capture) = vs.read()

    # break when no frame is grabbed (or end if the video)
    if not grabbed:
        print("[INFO] no capture read from stream - exiting")
        break

    # Check if the capture has valid dimensions
    if capture.shape[0] <= 0 or capture.shape[1] <= 0:
        print("[INFO] invalid frame dimensions - skipping")
        continue

    # resize frame and append it to our deque
    capture = cv2.resize(capture, dsize=(550, 400))
    captures.append(capture)

    # Process further only when the deque is filled
    if len(captures) < param.SAMPLE_DURATION:
        continue

    # now that our captures array is filled we can
    # construct our image blob
    # We will use SAMPLE_SIZE as height and width for
    # modifying the captured frame
    imageBlob = cv2.dnn.blobFromImages(captures, 1.0,
                                       (param.SAMPLE_SIZE,
                                        param.SAMPLE_SIZE),
                                       (114.7748, 107.7354, 99.4750),
                                       swapRB=True, crop=True)

    # Manipulate the image blob to make it fit as as input
    # for the pre-trained OpenCV's
    # Human Action Recognition Model
    imageBlob = np.transpose(imageBlob, (1, 0, 2, 3))
    imageBlob = np.expand_dims(imageBlob, axis=0)

    # Forward pass through model to make prediction
    net.setInput(imageBlob)
    outputs = net.forward()
    # Index the maximum probability
    label = param.CLASSES[np.argmax(outputs)]

    # Check if the detected action is "pushup"
    if label == "killing" and not pushup_detected:
        # Set the flag to True
        pushup_detected = True
        # Check cooldown time
        current_time = time.time()
        if current_time - last_alert_time >= param.COOLDOWN_TIME:
            # Speak the alert
            print("[INFO] Speaking alert: killing action detected")
            engine.say("killing action detected")
            engine.runAndWait()
            # Update last alert time
            last_alert_time = current_time

    # Reset the flag if another action is detected
    if label != "killing":
        pushup_detected = False

    # Show the predicted activity
    cv2.rectangle(capture, (0, 0), (300, 40), (255, 255, 255), -1)
    cv2.putText(capture, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 0), 2)

    # Display it on the screen
    cv2.imshow("Human Activity Recognition", capture)

    key = cv2.waitKey(1) & 0xFF
    # Press key 'q' to break the loop
    if key == ord("q"):
        break

# Release the video stream, close OpenCV windows, and stop the engine
vs.release()
cv2.destroyAllWindows()
engine.stop()
