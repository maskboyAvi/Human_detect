from ultralytics import YOLO
import cv2
import math

def live_video_detection():
    # Open webcam
    cap = cv2.VideoCapture(0)

    # Set up YOLO model
    model = YOLO('Model_Weights/final_best.pt')
    classNames = ["human"]

    while True:
        # Read frame from the webcam
        success, img = cap.read()

        # Perform object detection
        results = model(img, stream=True, conf=0.5)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(box.conf[0].item(), 2)
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name} {conf * 100}%'

                # Draw bounding box and label
                cv2.rectangle(img, (x1, y1), (x2, y2), (222, 51, 42), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (222, 51, 42), 2)

        # Show the frame
        cv2.imshow('YOLOv8 Object Detection', img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

# Call the function for live video detection
live_video_detection()
