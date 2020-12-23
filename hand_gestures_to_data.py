import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
from google.protobuf.json_format import MessageToDict
from datetime import datetime


class HandGesturesToData:
    """
    Class to use Google's Mediapipe HandTracking pipeline from Python.
    Get zoom video recode, divide each participants web cam input,
    detection of a hand and recognize the hand-finger gestures
    from 1 to 5.
    Args:
        video_path: video path of the zoom recording
        horiz_divisions:Number of tiles stacked horizontally
        vert_divisions:Number of tiles stacked vertically
    """

    def __init__(self, video_path, horiz_divisions, vert_divisions):
        self.video_path = video_path
        self.horiz_divisions = horiz_divisions
        self.vert_divisions = vert_divisions
        self.question_df = None
        self.df = None

    def __call__(self):
        """
         This method call split the video to division and data_analysis()
        """
        self.split_to_divisions()
        self.data_analysis()

    def show_plot_bar(self):
        """
        This method create bar plot from the data that analysis
        :return: bar plot figure
        """
        fig = None
        if self.question_df is not None:
            bar_plot = self.question_df.transpose().plot.bar(xlabel='Questions', ylabel='Answer', rot=0)
            fig = bar_plot.get_figure()
        return fig

    def save_data_csv(self):
        """
        This method create csv file of the result of data analysis
        example:
        "output_23_12_2020_11_23_44.csv"
        """
        now = datetime.now()
        current_time = now.strftime("%d_%m_%Y_%H_%M_%S")
        self.question_df.to_csv('output_' + current_time + ".csv")

    @staticmethod
    def image_processor(image):
        """
        This is static method
        get frame image and process the image with Google's Mediapipe,
        the result from  mp_hands.Hands contains
            -label right or left hand
            -hand landmark the represent 20 key points on the hand
            Example:
            #        8   12  16  20
            #        |   |   |   |
            #        7   11  15  19
            #    4   |   |   |   |
            #    |   6   10  14  18
            #    3   |   |   |   |
            #    |   5---9---13--17
            #    2    \         /
            #     \    \       /
            #      1    \     /
            #       \    \   /
            #        ------0-


        :param image:
        :return: Integer number of hand gesture if not find return 0
        """
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        try:
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
            if results.multi_hand_landmarks:
                image_rows, image_cols, _ = image.shape
                hand_direction = results.multi_handedness[0]
                hand_direction = MessageToDict(hand_direction)
                hand_direction = hand_direction['classification'][0]['label']
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    hand_landmark_dic = {}
                    # landmark contains 3 axis position (x,y,z)
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        hand_landmark_dic[idx] = landmark
                    if hand_direction == 'Right\r':
                        direction_right = True
                    if hand_direction == 'Left\r':
                        direction_right = False
                    # To know if the palm face in or out
                    # if the distance in x axis of the thumb from the start of the palm
                    if hand_landmark_dic[4].x < hand_landmark_dic[20].x and hand_direction == 'Right\r' or \
                            hand_landmark_dic[4].x > hand_landmark_dic[
                        20].x and hand_direction == 'Left\r':
                        palm_out = True
                    else:
                        palm_out = False

                    return HandGesturesToData.hand_gesture_calculate(hand_landmark_dic, direction_right, palm_out)
        finally:
            hands.close()

    @staticmethod
    def hand_gesture_calculate(hand_landmark_dic, direction_right, palm_out):
        """
        This static method calculate the hand gesture
        by the hand landmark, direction and palm position.
        by calculate the relative position of the top finger example(landmark[8],landmark[12])
        :param hand_landmark_dic: Dictionary of hand landmark key positions
        :param direction_right: Hand Direction left/right
        :param palm_out: if palm face out -True
        :return: Integer between 0-5
        """
        thumb_is_open = False
        first_finger_is_open = False
        second_finger_is_open = False
        third_finger_is_open = False
        fourth_finger_is_open = False
        pseudo_fix_key_point = hand_landmark_dic[2].x
        if palm_out and direction_right or not palm_out and not direction_right:
            if hand_landmark_dic[3].x < pseudo_fix_key_point and hand_landmark_dic[4].x < pseudo_fix_key_point:
                thumb_is_open = True
        else:
            if hand_landmark_dic[4].x > pseudo_fix_key_point and hand_landmark_dic[3].x > pseudo_fix_key_point:
                thumb_is_open = True
        pseudo_fix_key_point = hand_landmark_dic[6].y
        if hand_landmark_dic[7].y < pseudo_fix_key_point and hand_landmark_dic[8].y < pseudo_fix_key_point:
            first_finger_is_open = True
        pseudo_fix_key_point = hand_landmark_dic[10].y
        if hand_landmark_dic[11].y < pseudo_fix_key_point and hand_landmark_dic[12].y < pseudo_fix_key_point:
            second_finger_is_open = True
        pseudo_fix_key_point = hand_landmark_dic[14].y
        if hand_landmark_dic[15].y < pseudo_fix_key_point and hand_landmark_dic[16].y < pseudo_fix_key_point:
            third_finger_is_open = True
        pseudo_fix_key_point = hand_landmark_dic[18].y
        if hand_landmark_dic[19].y < pseudo_fix_key_point and hand_landmark_dic[20].y < pseudo_fix_key_point:
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

    def split_to_divisions(self):
        """ This method divide each participants web cam input"""
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        (h, w, d) = np.shape(frame)
        divisions = self.horiz_divisions * self.vert_divisions  # Total number of tiles
        data = [[] for i in range(divisions)]

        seg_h = int(h / self.vert_divisions)  # Tile height
        seg_w = int(w / self.horiz_divisions)  # Tile width

        while cap.isOpened():

            ret, frame = cap.read()
            if ret:
                vid = 0  # video counter
                for i in range(self.vert_divisions):
                    for j in range(self.horiz_divisions):
                        # Get the coordinates (top left corner) of the current tile
                        row = i * seg_h
                        col = j * seg_w
                        roi = frame[row:row + seg_h, col:col + seg_w, 0:3]  # Copy the region of interest
                        number = self.image_processor(roi)
                        if number is None:
                            number = 0
                        data[vid].append(number)
                        vid += 1
            else:
                break

        # Release all the objects
        cap.release()

        # store the all the data in pandas.DataFrame
        self.df = pd.DataFrame(data)
        # Release everything if job is finished
        cv2.destroyAllWindows()

    def data_analysis(self):
        """This method analysis the data that collect from the image processor.
            find when question ask and how many questions there was in the video
            depend on the second that all the participants votes
        """
        question_idx = []
        # add all the location where all the participants vote
        for column in self.df:
            # if in this frame/timestamp all the participants vote(hand gesture was recognized)
            if all(x > 0 for x in self.df[column]):
                question_idx.append(int(column))
        question_range = []
        # organize the location to range from start the question to the end
        start_idx = question_idx[0]
        for i in range(1, len(question_idx)):
            back = i - 1
            # 25-represent the number of frame per second on zoom recording video
            # bigger than 25 mean 1 second that no hand gesture was recognized
            if question_idx[i] - question_idx[back] >= 25:
                question_range.append((start_idx, question_idx[back]))
                start_idx = question_idx[i]
        question_range.append((start_idx, question_idx[-1]))
        index = 0
        column_name = ['question ' + str(idx + 1) for idx in range(len(question_range))]
        self.question_df = pd.DataFrame(columns=column_name, index=range(1, self.df.shape[0] + 1))
        for start_idx, end_idx in question_range:
            for ans_id in range(self.df.shape[0]):
                # check the max  values of the answer that recognized from participants in the rang
                # of the question
                # replace 0 to nan to find interesting hand gesture from the data
                temp = self.df.iloc[ans_id, start_idx:end_idx].replace(0, np.nan)
                if temp.isnull().values.all(axis=0):
                    # if all nan return zero
                    self.question_df.iloc[ans_id, index] = 0
                else:
                    self.question_df.iloc[ans_id, index] = temp.value_counts(dropna=True).idxmax()
            index += 1
