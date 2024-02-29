import cv2
import mediapipe as mp

def get_hand_point(hand_landmarks, point_index, img_shape):
    landmark = hand_landmarks.landmark[point_index]
    h, w, c = img_shape
    x, y = int(landmark.x * w), int(landmark.y * h)
    return x, y

def perform_actions(hand_points):

    p0x = hand_points[0][0]
    p1_1x= hand_points[1][0]
    p1_2x= hand_points[2][0]
    p1_3x= hand_points[3][0]
    p1_4x= hand_points[4][0]
    p2_1x= hand_points[5][0]
    p2_2x= hand_points[6][0]
    p2_3x= hand_points[7][0]
    p2_4x= hand_points[8][0]
    p2_1x= hand_points[9][0]
    p2_2x= hand_points[10][0]
    p2_3x= hand_points[11][0]
    p2_4x= hand_points[12][0]
    p2_1x= hand_points[13][0]
    p2_2x= hand_points[14][0]
    p2_3x= hand_points[15][0]
    p2_4x= hand_points[16][0]
    p2_1x= hand_points[17][0]
    p2_2x= hand_points[18][0]
    p2_3x= hand_points[19][0]
    p2_4x= hand_points[20][0]

    p0y = hand_points[0][1]
    p1_1y= hand_points[1][1]
    p1_2y= hand_points[2][1]
    p1_3y= hand_points[3][1]
    p1_4y= hand_points[4][1]
    p2_1y= hand_points[5][1]
    p2_2y= hand_points[6][1]
    p2_3y= hand_points[7][1]
    p2_4y= hand_points[8][1]
    p2_1y= hand_points[9][1]
    p2_2y= hand_points[10][1]
    p2_3y= hand_points[11][1]
    p2_4y= hand_points[12][1]
    p2_1y= hand_points[13][1]
    p2_2y= hand_points[14][1]
    p2_3y= hand_points[15][1]
    p2_4y= hand_points[16][1]
    p2_1y= hand_points[17][1]
    p2_2y= hand_points[18][1]
    p2_3y= hand_points[19][1]
    p2_4y= hand_points[20][1]


    # Example conditions for each hand point
    if 400 <= p0x <= 450 and 400 <= p0y <= 450:
        print("Condition for Point 0 is met!")
        # Add your specific action for this condition

    if 300 <= p1_1x <= 350 and 200 <= p1_1y <= 250:
        print("Condition for Point 1 is met!")
        # Add your specific action for this condition

    if 100 <= p1_2x <= 150 and 50 <= p1_2y <= 100:
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
