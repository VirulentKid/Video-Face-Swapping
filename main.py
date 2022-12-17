import cv2
import numpy as np
import pdb
import sys
import matplotlib.pyplot as plt
import traceback
import time
from helper import *
import landmark_detection
from convex_hull import convex_hull
from triangulation import triangulation
from warping import warping
from cloning import cloning
from cartoon import *
from opticalFlow import *

SOURCE_PATH = 'videos/LucianoRosso1.mp4'
TARGET_PATH = 'videos/FrankUnderwood.mp4'
SWAP_RATE = 5
TARGET_EMOTION = True
CARTOON = False

if __name__ == "__main__":
    # read source video
    vid_s = cv2.VideoCapture(SOURCE_PATH)
    # read target video
    vid_t = cv2.VideoCapture(TARGET_PATH)

    # reference link: enum: VideoCapture generic properties identifier
    # get CAP_PROP_POS_FRAMES: 0-based index of the frame to be decoded/captured next.
    pos_frame_t = vid_t.get(1)
    width = int(vid_t.get(3))
    height = int(vid_t.get(4))

    # result should have the same width and height with the target video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    res = cv2.VideoWriter('res.avi', fourcc, 24, (width, height))

    # limit = 1000
    start = time.time()

    # jump = 137
    pt_copy = []
    while True:
        # keep read in
        ret_s, frame_s = vid_s.read()
        ret_t, frame_t = vid_t.read()
        read_in_flag = ret_s and ret_t

        # wrapped_t: the canvas that will be wrapped and clone to the target
        wrapped_t = np.copy(frame_t)
        p_s = []  # initialize axis of source image's landmark
        p_t = []  # initialize axis of target image's landmark

        if read_in_flag:
            pos_frame_t = vid_t.get(1)
            print(pos_frame_t)
            if pos_frame_t % SWAP_RATE == 1:  # point to swap face
                # 1: Detect face (assume single face in each video) and landmarks(eyes, noses, etc.)
                try:
                    p_s, p_t, landmarks_s, landmarks_t = landmark_detection.landmark_detect_c(
                        frame_s, frame_t)
                    pt_copy = p_t
                except:
                    if empty_pts(p_s, p_t):
                        print("pass because empty")
                        continue
                # print(p_t)
                #feature_visualize(frame_s, p_s)
                #feature_visualize(frame_t, p_t)

                # 2: Convex Hull to detect face boundary
                hull_s = []
                hull_t = []
                try:
                    if TARGET_EMOTION:
                        hull_s, hull_t = convex_hull_target_emotion(
                            p_s, p_t, landmarks_s, landmarks_t)
                    else:
                        hull_s, hull_t = convex_hull(
                            p_s, p_t, landmarks_s, landmarks_t)
                except:
                    if empty_pts(hull_s, hull_t):
                        continue

                # feature_visualize(frame_s, hull_s)
                # feature_visualize(frame_t, hull_t)

            #     # STEP 3: Triangulation
                hull_t = np.asarray(hull_t)
                frame_t_width = frame_t.shape[1]
                frame_t_height = frame_t.shape[0]
                hull_t[:, 0] = np.clip(
                    hull_t[:, 0], 0, frame_t_width - 1)
                hull_t[:, 1] = np.clip(
                    hull_t[:, 1], 0, frame_t_height - 1)
                temp = []
                hull_t = hull_t.astype(np.int32).tolist()
                hull_t = tupleList(temp, hull_t)
                tri_result = triangulation(frame_t, hull_t)
                if len(tri_result) == 0:
                    continue

            #     # STEP 4: Warping
                warping(tri_result, hull_s, hull_t, frame_s, wrapped_t)

                # STEP 5: Cloning
                output = cloning(wrapped_t, frame_t, hull_t)
                if CARTOON:
                    cartoon_output = cartoon_effect(output)
                    res.write(cartoon_output)
                else:
                    res.write(output)

                prev_frame_t = frame_t

            else:
                try:
                    output, pt_copy = performOpticalFlow(
                        output, pt_copy, frame_t, prev_frame_t)
                    if CARTOON:
                        cartoon_output = cartoon_effect(output)
                        res.write(cartoon_output)
                    else:
                        res.write(output)
                    prev_frame_t = frame_t
                except KeyboardInterrupt:
                    sys.exit()
                except:
                    print(traceback.format_exc())
                    continue

        if cv2.waitKey(10) == 27 or vid_t.get(cv2.CAP_PROP_POS_FRAMES) == vid_t.get(cv2.CAP_PROP_FRAME_COUNT)\
                or vid_s.get(cv2.CAP_PROP_POS_FRAMES) == vid_t.get(cv2.CAP_PROP_FRAME_COUNT):
            break

    print('time taken :' + str(time.time() - start))
    cv2.destroyAllWindows()
    vid_s.release()
    vid_t.release()
    res.release()
