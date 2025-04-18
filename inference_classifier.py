import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="asl_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.9)

# Label map — same as before
labels_dict = {0: 'A', 1: 'E', 2: 'I', 3: 'O', 4: 'U'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Collect landmark positions
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Bounding box
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # Prepare input for TFLite model
            input_data = np.array([data_aux], dtype=np.float32)

            try:
                # Set tensor
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                # Get prediction
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # batch size 1
                predicted_index = np.argmax(output_data)
                confidence = np.max(output_data)

                if confidence >= 0.8:
                    predicted_character = labels_dict.get(predicted_index, "Gesture incorrect")
                    print(f"Predicted Gesture: {predicted_character} ({confidence*100:.2f}%)")
                else:
                    predicted_character = "Gesture incorrect"
                    print("Gesture confidence too low. Marking as incorrect.")

                # Draw result
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            except Exception as e:
                print("Prediction error:", e)
                cv2.putText(frame, "Gesture incorrect", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

    else:
        print("No hand detected.")
        cv2.putText(frame, "No hand detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
