import cv2
import mediapipe as mp
from flask import Flask, Response, jsonify, send_from_directory, request
import numpy as np
import random
import os
import time

app = Flask(__name__)

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

finger_tips = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

# Game state
player_score = 0
computer_score = 0
round_num = 0
last_round_time = 0
current_player_gesture = "👀"
current_computer_gesture = "🤖"
round_result = ""

computer_choices = [
    {"name": "Rock 👊", "value": 0, "emoji": "👊"},
    {"name": "Scissors ✌️", "value": 2, "emoji": "✌️"},
    {"name": "Paper ✋", "value": 5, "emoji": "✋"}
]

def get_gesture(finger_count):
    gestures = {
        0: "Rock 👊",
        2: "Scissors ✌️",
        5: "Paper ✋"
    }
    return gestures.get(finger_count, "Unknown Gesture")

def determine_winner(player_count, computer_count):
    global player_score, computer_score, round_result
    if player_count == computer_count:
        round_result = "It's a Tie!"
        return "Tie"
    if (player_count == 0 and computer_count == 2) or \
       (player_count == 2 and computer_count == 5) or \
       (player_count == 5 and computer_count == 0):
        player_score += 1
        round_result = "Player Wins This Round!"
        return "Player Wins!"
    else:
        computer_score += 1
        round_result = "Computer Wins This Round!"
        return "Computer Wins!"

def generate_frames():
    global round_num, last_round_time, current_player_gesture, current_computer_gesture
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        finger_count = 0
        gesture = "No Hand Detected"
        computer_gesture = ""
        result = ""

        current_time = time.time()

        if round_num < 5 and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark

                if landmarks[finger_tips[0]].x > landmarks[finger_tips[0] - 1].x:
                    finger_count += 1
                for i in range(1, 5):
                    if landmarks[finger_tips[i]].y < landmarks[finger_tips[i] - 2].y:
                        finger_count += 1

                gesture = get_gesture(finger_count)
                
                # Only process a new round every 3 seconds
                if gesture != "Unknown Gesture" and current_time - last_round_time >= 3:
                    computer_choice = random.choice(computer_choices)
                    computer_gesture = computer_choice["name"]
                    
                    # Update game state
                    round_num += 1
                    last_round_time = current_time
                    
                    # Update current gestures for display
                    current_player_gesture = gesture
                    current_computer_gesture = computer_choice["emoji"]
                    
                    # Determine winner
                    result = determine_winner(finger_count, computer_choice["value"])

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/game_state')
def game_state():
    return jsonify({
        'player_score': player_score,
        'computer_score': computer_score,
        'round': round_num,
        'game_over': round_num >= 5,
        'player_gesture': current_player_gesture,
        'computer_gesture': current_computer_gesture,
        'round_result': round_result
    })

@app.route('/reset_game', methods=['POST'])
def reset_game():
    global player_score, computer_score, round_num, current_player_gesture, current_computer_gesture, round_result
    player_score = 0
    computer_score = 0
    round_num = 0
    current_player_gesture = "👀"
    current_computer_gesture = "🤖"
    round_result = ""
    return jsonify({"status": "Game Reset"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)