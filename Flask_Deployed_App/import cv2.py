import cv2
import mediapipe as mp
import pyautogui

# Get screen size
screen_width, screen_height = pyautogui.size()

# OpenCV camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Smooth cursor
prev_x, prev_y = 0, 0
smoothening = 7

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # RGB image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index fingertip position (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]
            x = int(index_finger_tip.x * frame_width)
            y = int(index_finger_tip.y * frame_height)

            # Convert to screen coordinates
            screen_x = screen_width * (x / frame_width)
            screen_y = screen_height * (y / frame_height)

            # Smooth the movement
            cur_x = prev_x + (screen_x - prev_x) / smoothening
            cur_y = prev_y + (screen_y - prev_y) / smoothening

            # Move mouse
            pyautogui.moveTo(cur_x, cur_y)
            prev_x, prev_y = cur_x, cur_y

            # Optional: draw pointer circle
            cv2.circle(frame, (x, y), 10, (255, 0, 0), cv2.FILLED)

    cv2.imshow("Virtual Hand Mouse", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
