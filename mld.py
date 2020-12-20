import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import pandas as pd
from google.protobuf.json_format import MessageToDict


def hand_pro(image):
    landmark_pb2.NormalizedLandmarkList
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    try:
        # For static images:
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7)
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imshow('MediaPipe Hands', image)
        if results.multi_hand_landmarks:
            image_rows, image_cols, _ = image.shape
            hand_direction = results.multi_handedness[0]
            hand_direction = MessageToDict(hand_direction)
            hand_direction = hand_direction['classification'][0]['label']
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                dic = {}
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    dic[idx] = landmark
                if hand_direction == 'Right\r':
                    direction_right = True
                if hand_direction == 'Left\r':
                    direction_right = False
                if dic[4].x < dic[20].x and hand_direction == 'Right\r' or dic[4].x > dic[
                    20].x and hand_direction == 'Left\r':
                    palm_out = True
                else:
                    palm_out = False

                color = (255, 0, 0)

                # Line thickness of 2 px
                thickness = 2

                thumb_is_open = False
                first_finger_is_open = False
                second_finger_is_open = False
                third_finger_is_open = False
                fourth_finger_is_open = False

                pseudo_fix_key_point = dic[2].x
                if palm_out and direction_right or not palm_out and not direction_right:
                    if dic[3].x < pseudo_fix_key_point and dic[4].x < pseudo_fix_key_point:
                        thumb_is_open = True
                else:
                    if dic[4].x > pseudo_fix_key_point and dic[3].x > pseudo_fix_key_point:
                        thumb_is_open = True
                pseudo_fix_key_point = dic[6].y
                if dic[7].y < pseudo_fix_key_point and dic[8].y < pseudo_fix_key_point:
                    first_finger_is_open = True
                pseudo_fix_key_point = dic[10].y
                if dic[11].y < pseudo_fix_key_point and dic[12].y < pseudo_fix_key_point:
                    second_finger_is_open = True
                pseudo_fix_key_point = dic[14].y
                if dic[15].y < pseudo_fix_key_point and dic[16].y < pseudo_fix_key_point:
                    third_finger_is_open = True
                pseudo_fix_key_point = dic[18].y
                if dic[19].y < pseudo_fix_key_point and dic[20].y < pseudo_fix_key_point:
                    fourth_finger_is_open = True

                if thumb_is_open and first_finger_is_open and second_finger_is_open and third_finger_is_open and fourth_finger_is_open:
                    return 5
                elif not thumb_is_open and first_finger_is_open and second_finger_is_open and third_finger_is_open and fourth_finger_is_open:
                    return 4
                elif not thumb_is_open and first_finger_is_open and second_finger_is_open and third_finger_is_open and not fourth_finger_is_open:
                    return 3
                elif not thumb_is_open and first_finger_is_open and second_finger_is_open and not third_finger_is_open and not fourth_finger_is_open:
                    return 2
                elif not thumb_is_open and first_finger_is_open and not second_finger_is_open and not third_finger_is_open and not fourth_finger_is_open:
                    return 1
                else:
                    return 0
    finally:
        hands.close()

# landmark_pb2.NormalizedLandmarkList
# mp_drawing = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands
# data = pd.DataFrame()
#
# # For static images:
# hands = mp_hands.Hands(
#     static_image_mode=True,
#     max_num_hands=2,
#     min_detection_confidence=0.7)
# for idx, file in enumerate(file_list):
#   # Read an image, flip it around y-axis for correct handedness output (see
#   # above).
#   image = cv2.flip(cv2.imread(file), 1)
#   # Convert the BGR image to RGB before processing.
#   results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#
#   # Print handedness and draw hand landmarks on the image.
#   print('handedness:', results.multi_handedness)
#   if not results.multi_hand_landmarks:
#     continue
#   annotated_image = image.copy()
#   for hand_landmarks in results.multi_hand_landmarks:
#     print('hand_landmarks:', hand_landmarks)
#     mp_drawing.draw_landmarks(
#         annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#   cv2.imwrite(
#       '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(image, 1))
# hands.close()

# For webcam input:
# hands = mp_hands.Hands(
#     min_detection_confidence=0.7, min_tracking_confidence=0.5,max_num_hands=8)
#
# cap = cv2.VideoCapture('zoom_0.mp4')
# while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#         break
#
#     # Flip the image horizontally for a later selfie-view display, and convert
#     # the BGR image to RGB.
#     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     results = hands.process(image)
#
#     # Draw the hand annotations on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#     if results.multi_hand_landmarks:
#         image_rows, image_cols, _ = image.shape
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             # print(hand_landmarks)
#             dic = {}
#             for idx, landmark in enumerate(hand_landmarks.landmark):
#                 dic[idx] = landmark
#
#             color = (255, 0, 0)
#
#             # Line thickness of 2 px
#             thickness = 2
#
#             thumbIsOpen = False
#             firstFingerIsOpen = False
#             secondFingerIsOpen = False
#             thirdFingerIsOpen = False
#             fourthFingerIsOpen = False
#
#             pseudoFixKeyPoint = dic[2].x
#             if dic[3].x < pseudoFixKeyPoint and dic[4].x < pseudoFixKeyPoint:
#                 thumbIsOpen = True
#             pseudoFixKeyPoint = dic[6].y
#             if dic[7].y < pseudoFixKeyPoint and dic[8].y < pseudoFixKeyPoint:
#                 firstFingerIsOpen = True
#             pseudoFixKeyPoint = dic[10].y
#             if dic[11].y < pseudoFixKeyPoint and dic[12].y < pseudoFixKeyPoint:
#                 secondFingerIsOpen = True
#             pseudoFixKeyPoint = dic[14].y
#             if dic[15].y < pseudoFixKeyPoint and dic[16].y < pseudoFixKeyPoint:
#                 thirdFingerIsOpen = True
#             pseudoFixKeyPoint = dic[18].y
#             if dic[19].y < pseudoFixKeyPoint and dic[20].y < pseudoFixKeyPoint:
#                 fourthFingerIsOpen = True
#
#             if thumbIsOpen and firstFingerIsOpen and secondFingerIsOpen and thirdFingerIsOpen and fourthFingerIsOpen:
#                 print('FIVE')
#
#             elif not thumbIsOpen and firstFingerIsOpen and secondFingerIsOpen and thirdFingerIsOpen and fourthFingerIsOpen:
#                 print('FOUR')
#             elif not thumbIsOpen and firstFingerIsOpen and secondFingerIsOpen and thirdFingerIsOpen and not fourthFingerIsOpen:
#                 print('THREE')
#             elif not thumbIsOpen and firstFingerIsOpen and secondFingerIsOpen and not thirdFingerIsOpen and not fourthFingerIsOpen:
#                 print('TWO')
#             elif not thumbIsOpen and firstFingerIsOpen and not secondFingerIsOpen and not thirdFingerIsOpen and not fourthFingerIsOpen:
#                 print('ONE')
#
#         cv2.imshow('MediaPipe Hands', image)
#     if cv2.waitKey(5) & 0xFF == 27:
#         break
# hands.close()
# cap.release()
