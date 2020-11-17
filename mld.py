import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
landmark_pb2.NormalizedLandmarkList
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For static images:
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7)
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
hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
while cap.isOpened():
  success, image = cap.read()
  if not success:
    break

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
  if results.multi_hand_landmarks:
    image_rows, image_cols, _ = image.shape
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
          image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        dic={}
        for idx, landmark in enumerate(hand_landmarks.landmark):
             dic[idx]=landmark
        # max_x = max(dic.values(),key=lambda p:p.x)
        # max_y = max(dic.values(),key=lambda p:p.y)
        # min_x = min(dic.values(),key=lambda p:p.x)
        # min_y = min(dic.values(),key=lambda p:p.y)
        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2


        thumbIsOpen = False
        firstFingerIsOpen = False
        secondFingerIsOpen = False
        thirdFingerIsOpen = False
        fourthFingerIsOpen = False

        pseudoFixKeyPoint = dic[2].x
        if dic[3].x < pseudoFixKeyPoint and dic[4].x < pseudoFixKeyPoint:
            thumbIsOpen = True
        pseudoFixKeyPoint = dic[6].y
        if dic[7].y < pseudoFixKeyPoint and dic[8].y < pseudoFixKeyPoint:
                firstFingerIsOpen = True
        pseudoFixKeyPoint = dic[10].y
        if dic[11].y < pseudoFixKeyPoint and dic[12].y < pseudoFixKeyPoint:
                secondFingerIsOpen = True
        pseudoFixKeyPoint = dic[14].y
        if dic[15].y < pseudoFixKeyPoint and dic[16].y < pseudoFixKeyPoint:
                thirdFingerIsOpen = True
        pseudoFixKeyPoint = dic[18].y
        if dic[19].y < pseudoFixKeyPoint and dic[20].y < pseudoFixKeyPoint:
                fourthFingerIsOpen = True


        if thumbIsOpen and firstFingerIsOpen and secondFingerIsOpen and thirdFingerIsOpen and fourthFingerIsOpen:
            print('FIVE')

        elif not thumbIsOpen and firstFingerIsOpen and secondFingerIsOpen and thirdFingerIsOpen and fourthFingerIsOpen:
            print('FOUR')
        elif thumbIsOpen and firstFingerIsOpen and secondFingerIsOpen and not thirdFingerIsOpen and not fourthFingerIsOpen:
            print('THREE')
        elif thumbIsOpen and firstFingerIsOpen and not secondFingerIsOpen and not thirdFingerIsOpen and not fourthFingerIsOpen:
            print('TWO')
        elif not thumbIsOpen and firstFingerIsOpen and not secondFingerIsOpen and not thirdFingerIsOpen and not fourthFingerIsOpen:
            print('ONE')

    cv2.imshow('MediaPipe Hands', image)
  if cv2.waitKey(5) & 0xFF == 27:
    break
hands.close()
cap.release()