import numpy as np
import cv2
import mediapipe as mp

import pandas as pd
from google.protobuf.json_format import MessageToDict
import matplotlib.pyplot as plt


def hand_pro(image):
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


def split_to_divisions(video_path, horiz_divisions, vert_divisions):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    (h, w, d) = np.shape(frame)
    divisions = horiz_divisions * vert_divisions  # Total number of tiles
    data = [[] for i in range(divisions)]

    # horiz_divisions = 2  # Number of tiles stacked horizontally
    # vert_divisions = 1  # Number of tiles stacked vertically

    seg_h = int(h / vert_divisions)  # Tile height
    seg_w = int(w / horiz_divisions)  # Tile width

    # Initialise the output videos
    # out_videos = [0] * divisions

    # for i in range(divisions):
    #     out_videos[i] = cv2.VideoWriter('out{}.avi'.format(str(i)), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
    #                                    (seg_w, seg_h))

    # main code
    while (cap.isOpened()):

        ret, frame = cap.read()
        if ret == True:
            vid = 0  # video counter
            for i in range(vert_divisions):
                for j in range(horiz_divisions):
                    # Get the coordinates (top left corner) of the current tile
                    row = i * seg_h
                    col = j * seg_w
                    roi = frame[row:row + seg_h, col:col + seg_w, 0:3]  # Copy the region of interest
                    # out_videos[vid].write(roi)

                    number = hand_pro(roi)
                    if number is None:
                        number = 0
                    data[vid].append(number)
                    vid += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release all the objects
    cap.release()
    # for i in range(divisions):
    #     out_videos[i].release()
    # Release everything if job is finished
    array = np.array(data, dtype=int)
    df = pd.DataFrame(array)
    # df.to_csv('file2.csv', index=False)
    cv2.destroyAllWindows()
    return df


def data_analysis(df):
    question_idx = []

    for column in df:
        if all(x > 0 for x in df[column]):
            question_idx.append(int(column))
    ans = []
    start_idx = question_idx[0]
    for i in range(1, len(question_idx)):
        back = i - 1
        if question_idx[i] - question_idx[back] >= 25:
            ans.append((start_idx, question_idx[back]))
            start_idx = question_idx[i]
    ans.append((start_idx, question_idx[-1]))
    index = 0
    column_name = ['question ' + str(idx + 1) for idx in range(len(ans))]
    question_df = pd.DataFrame(columns=column_name, index=range(1, df.shape[0] + 1))
    for start_idx, end_idx in ans:
        for ans_id in range(df.shape[0]):
            question_df.iloc[ans_id, index] = df.iloc[ans_id, start_idx:end_idx].replace(0, np.nan).value_counts(
                dropna=True).idxmax()
        index += 1
    question_df.transpose().plot.bar(xlabel='Participants number', ylabel='Answer')
    # sns.barplot(data=question_df)

    plt.show()


data_from_video = split_to_divisions('zoom_1.mp4', 2, 1)
data_analysis(data_from_video)