import cv2
import mediapipe as mp

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Finger tip landmark IDs (Thumb, Index, Middle, Ring, Pinky)
finger_tips = [4, 8, 12, 16, 20]

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for better user experience
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    finger_count = 0  # Variable to store the number of raised fingers

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark

            # Thumb: Check if it's to the right of the lower thumb joint
            if landmarks[finger_tips[0]].x > landmarks[finger_tips[0] - 1].x:
                finger_count += 1

            # Other fingers: Check if tip is above the second knuckle
            for i in range(1, 5):
                if landmarks[finger_tips[i]].y < landmarks[finger_tips[i] - 2].y:
                    finger_count += 1

    # Display the detected number
    cv2.putText(frame, f"Number: {finger_count}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Hand Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
