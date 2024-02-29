import cv2
import mediapipe as mp

def get_hand_point(hand_landmarks, point_index, img_shape):
    landmark = hand_landmarks.landmark[point_index]
    h, w, c = img_shape
    x, y = int(landmark.x * w), int(landmark.y * h)
    return x, y

def perform_actions(hand_points):
    # Example conditions for each hand point
    if 400 <= hand_points[0][0] <= 450 and 400 <= hand_points[0][1] <= 450:
        print("Condition for Point 0 is met!")
        # Add your specific action for this condition

    if 300 <= hand_points[1][0] <= 350 and 200 <= hand_points[1][1] <= 250:
        print("Condition for Point 1 is met!")
        # Add your specific action for this condition

    if 100 <= hand_points[2][0] <= 150 and 50 <= hand_points[2][1] <= 100:
        print("Condition for Point 2 is met!")
        # Add your specific action for this condition

    # Add conditions for other hand points...

# Initialize variables
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

display_hand_points = True  # Boolean flag to control displaying hand points
display_detection_lines = True  # Boolean flag to control displaying hand detection lines

hand_points_str_top = ""  # String to accumulate hand point positions (top line)
hand_points_str_bottom = ""  # String to accumulate hand point positions (bottom line)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    hand_points = {}  # Dictionary to store hand point positions

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                point_position = get_hand_point(handLms, id, img.shape)
                hand_points[id] = point_position

            if display_detection_lines:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Call the function to perform actions based on hand point positions
            perform_actions(hand_points)

    if display_hand_points:
        # Accumulate hand point positions in separate strings for top and bottom lines
        hand_points_str_top = "\n".join([f"P{id}: {point}" for id, point in hand_points.items() if id < 8])
        hand_points_str_bottom = "\n".join([f"P{id}: {point}" for id, point in hand_points.items() if 15 > id >= 8])
        hand_points_str_bottom1 = "\n".join([f"P{id}: {point}" for id, point in hand_points.items() if id >= 15])

        # Display hand point positions on two lines with a smaller font
        cv2.putText(img, hand_points_str_top, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(img, hand_points_str_bottom, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(img, hand_points_str_bottom1, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Resize the img for a larger display
    img = cv2.resize(img, (int(img.shape[1] * 1.5), int(img.shape[0] * 1.5)))

    cv2.imshow("image", img)

    # Check for the 'q' key to exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):  # Toggle displaying hand points on 'd' key press
        display_hand_points = not display_hand_points
    elif key == ord('p'):  # Toggle displaying hand detection lines on 'p' key press
        display_detection_lines = not display_detection_lines

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
