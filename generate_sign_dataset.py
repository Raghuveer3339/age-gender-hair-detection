import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# FIX 1: Force camera backend (IMPORTANT)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# FIX 2: Create window properly
cv2.namedWindow("Collecting Data", cv2.WINDOW_NORMAL)

# Ask for label
label = input("Enter label (Hello / Yes / No / Thanks): ")

# Open CSV file
with open("sign_data.csv", "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera not working")
            break

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract features
                features = []
                for lm in hand_landmarks.landmark:
                    features.append(lm.x)
                    features.append(lm.y)

                # Save row
                row = [label] + features
                writer.writerow(row)

        # Show frame (VERY IMPORTANT)
        cv2.imshow("Collecting Data", frame)

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()