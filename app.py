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

def get_gesture(finger_count):
    """
    Determine the gesture based on finger count
    """
    gestures = {
        0: "Rock 👊",
        2: "Scissors ✌️",
        5: "Paper ✋"
    }
    return gestures.get(finger_count, "Unknown Gesture")

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
    gesture = "No Hand Detected"

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

            # Get the gesture based on finger count
            gesture = get_gesture(finger_count)

    # Display the detected number and gesture
    cv2.putText(frame, f"Number: {finger_count}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    cv2.putText(frame, f"Gesture: {gesture}", (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    cv2.imshow("Hand Gesture Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()