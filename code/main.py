import cv2
import numpy as np
from scipy.spatial import distance
import math
import os

def show_wrap(name, img, size, close=True, video=False):
    """
    :param name: [string] - Name of the window to be created
    :param img: [np.ndarray] - Image to be shown
    :param size: [int] - Resizing of the window
    :param close: [bool] - If 'False' window stays on screen (for comparisons)
    :param video: [bool] - If 'True' window tracks 'esc' as input for end of the displaying
    :return: null
    Simple wrapper for cv2.imshow(). Resize with original aspect ratio, display, wait for keystroke and close.
    """
    k = 0
    h, w = img.shape[:2]
    aspect_ratio = min(w, h) / max(w, h)
    img_res = cv2.resize(img, (size, np.intp(size * aspect_ratio)))
    cv2.imshow(name, img_res)
    if video:
        k = cv2.waitKey(1) & 0xff
    else:
        cv2.waitKey(0)
    if close:
        cv2.destroyAllWindows()
    return k


def morphology(bag_seg):
    """
    :param bag_seg: [np.ndarray] - Result of background subtraction
    :return: [np.ndarray] - Image changed by morphological operations
    This function simplifies the binary image returned by background subtraction.
    Aim is to:
     1) suppress noise (MORPH_OPEN with small kernel)
     2) enlarge foreground objects (Dilate with bigger ellipse kernel)
     3) close gaps in objects for future contour look-up (MORPH_CLOSE with bigger kernel)
    """
    ret, img_thresh = cv2.threshold(bag_seg, 254, 255, cv2.THRESH_BINARY)
    kernel_1 = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_1)
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilation = cv2.dilate(opening, kernel_2, iterations=1)
    kernel_3 = np.ones((11, 11), np.uint8)
    close = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_3)
    return close


def detect_ball(contours):
    """
    :param contours: [np.ndarray] - list of contours
    :return: [np.ndarray] - list of eligible center-points
    This function filters through contour list and returns those that could fit the shape of a tennis ball
    """
    min_radius_thresh= 5
    max_radius_thresh= 30
    centers=[]

    for c in contours:
        (x, y), radius = cv2.minEnclosingCircle(c)
        radius = int(radius)
        if (radius > min_radius_thresh) and (radius < max_radius_thresh):
            centers.append(np.array([[x], [y]]))
    return centers


def distance_rect_p(bb, p):
    """
    :param bb: [np.array] - bounding box (of a player)
    :param p: [tuple] - point (center of a ball)
    :return: [float] - distance between point and bounding box
    """
    dx = max(bb[0] - p[0], 0, p[0] - bb[0] - bb[2])
    dy = max(bb[1] - p[1], 0, p[1] - bb[1]- bb[3])
    return math.sqrt(dx*dx + dy*dy)


def ball_tracking(result, cnts, prev_pos, player_1_bb, player_2_bb, near_1, near_2, entry_pos):
    """
    :param result: [np.ndarray] - Image for drawing the results
    :param cnts: [np.ndarray] - list of contours
    :param prev_pos: [tuple] - previous position of the ball
    :param player_1_bb: [np.array] - bounding box (of player 1)
    :param player_2_bb: [np.array] - bounding box (of player 2)
    :param near_1: [bool] - True if the ball was close to the player 1 in last frame
    :param near_2: [bool] - True if the ball was close to the player 2 in last frame
    :param entry_pos: [tuple] - position of the ball when it first got close to a player
    :return: [np.ndarray], [tuple], [bool], [bool], [tuple] - viz. param descriptions
    Tracking of ball based on three criteria:
     1) closeness to center of the court
     2) closeness to previous position of the ball
     3) closeness to last entry position of the ball
    Function starts with detection of potential ball positions and checks if previous position of the ball was not in
    proximity of a player. If so, adequate switches are changed and main decision tree begins.
    If no eligible center of contour was found, the function ends and passes back the parameters.
    If there is at least one center, then the decision is split few times.
    First, the 'distance of eligible centers to previous point' is initiated. Either the previous position of ball
     does not exist => initialize with zeroes;
     or we know previous position => calculate distances to eligible centers.
    Second, similarly to first decision the 'distance of eligible centers to entry point' is initialized, but
     if the entry position doe not exist we initialize with large number.
    Third, the decision which center is best fit to be the ball is made:
     Either one of the centers is obviously close to the previous position => it should be a ball;
     Or the ball was probably lost for a short time because it was obscured by a player => ball will be close to the
      entry position; [both 'Either' and 'Or' decisions are summed with distances to middle of the court]
     (Else any position of eligible centers did not fit the criteria => the ball is temporarily lost and we will not
      update the position.)
    Fourth, different color is applied based on closeness to players.
    Then if current position is suficiently far from entry position the player probably hit it => null 'entry_position
     and 'near_' switches
    """
    x_img_center = 960
    y_img_center = 400
    centers = detect_ball(cnts)
    if distance_rect_p(player_1_bb, prev_pos) < 60:
        if not near_1:
            entry_pos = prev_pos
        near_1 = True
    elif distance_rect_p(player_2_bb, prev_pos) < 60:
        if not near_2:
            entry_pos = prev_pos
        near_2 = True
    if len(centers) > 0:
        tmp = []
        for c in centers:
            tmp.append((int(c[0]), int(c[1])))
            cv2.circle(result, (int(c[0]), int(c[1])), 10, (0, 0, 255), 2)
        dist_to_mid = distance.cdist(np.array([(x_img_center, y_img_center)]), tmp)
        if prev_pos[0] != prev_pos[1] != 0:
            dist_to_prev = distance.cdist(np.array([prev_pos]), tmp)
        else:
            dist_to_prev = np.array([0] * len(tmp))
        if entry_pos[0] != 0 and entry_pos[1] != 0:
            dist_to_entry = distance.cdist(np.array([entry_pos]), tmp)
        else:
            dist_to_entry = np.array([10000] * len(tmp))
        if (prev_pos[0] == 0 and prev_pos[1] == 0) or dist_to_prev.min() < 150:
            id = np.array([a + b for a, b in zip(dist_to_mid, dist_to_prev)]).argmin()
            prev_pos = (int(tmp[id][0]), int(tmp[id][1]))
        elif dist_to_entry.min() < 250:
           id = np.array([a + b for a, b in zip(dist_to_mid, dist_to_entry)]).argmin()
           prev_pos = (int(tmp[id][0]), int(tmp[id][1]))
        if near_1:
            cv2.circle(result, prev_pos, 10, (255, 255, 0), 2)
        elif near_2:
            cv2.circle(result, prev_pos, 10, (255, 0, 255), 2)
        else:
            cv2.circle(result, prev_pos, 10, (0, 255, 255), 2)
        if math.dist(entry_pos, prev_pos) > 110:
            near_1 = False
            near_2 = False
            entry_pos = (0, 0)
    return result, prev_pos, near_1, near_2, entry_pos


def m_s_setup(frame, cnt):
    """
    :param frame: [np.ndarray] - Original image for set-up of meanshift algorithm
    :param cnt: [np.array] - contour of a player
    :return: [np.array],[tuple], [numpy.ndarray] - parameters for meanshift algorithm
    Set-up of parameters for meanshift algorithm
    """
    x, y, w, h = cv2.boundingRect(cnt)
    track_window = (x, y, w, h)
    roi = frame[y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((5, 20, 20)), np.array((180, 250, 250)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    return track_window, term_crit, roi_hist


def apply_mask(morp_f, mask_r):
    """
    :param morp_f: [np.ndarray] - Binary image of background subtraction (with morphological filters applied)
    :param mask_r: [np.ndarray] - Image to be changed to HSV, masked and returned
    :return: [np.ndarray] - masked image (mask_r)
    Change to HSV, apply background subtraction mask, return
    """
    m_f_stack = np.dstack([morp_f] * 3)
    m_f_stack = m_f_stack.astype('float32') / 255.0
    mask_r = mask_r.astype('float32') / 255.0
    mask = (m_f_stack * mask_r) + ((1 - m_f_stack) * (0.0, 0.0, 0.0))
    return (mask * 255).astype('uint8')


def show_hits(res, n_1, n_2, switch, score):
    """
    :param res: [np.ndarray] - Image for drawing of results
    :param n_1: [bool] - 'True' if near player 1
    :param n_2: [bool] - 'True' if near player 2
    :param switch: [integer] - '0' if nobody touched the ball yet, '1' or '2' depending on last player that was near
    :param score: [tuple] - current score of hits
    :return: [np.ndarray], [integer], [tuple] - viz. param description
    Put text of current score on the image and add score based on closeness to the players
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    color = (255, 0, 0)
    thickness = 2

    if n_1 and switch != 1 and not n_2:
        score[0] += 1
        switch = 1
    elif n_2 and switch != 2 and not n_1:
        score[1] += 1
        switch = 2
    cv2.putText(res, "HITS: ", (1680, 800), font, scale, color, thickness)
    cv2.putText(res, "PLAYER_1 - " + str(score[0]), (1680, 850), font, scale, color, thickness)
    cv2.putText(res, "PLAYER_2 - " + str(score[1]), (1680, 900), font, scale, color, thickness)
    return res, switch, score

def show_win(res, switch):
    """
    :param res: [np.ndarray] - Image for drawing of results
    :param switch: [integer] - '0' if nobody touched the ball yet, '1' or '2' depending on last player that was near
    :return: null
    Display last read frame of the video with information about the winner based on the last player to be near the ball.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2
    color = (0, 255, 255)
    thickness = 3
    cv2.putText(res, "WINNER", (850, 300), font, scale, color, thickness)
    if switch == 1:
        cv2.putText(res, "--PLAYER 1--", (720, 400), font, scale, color, thickness)
    elif switch == 2:
        cv2.putText(res, "--PLAYER 2--", (720, 400), font, scale, color, thickness)
    else:
        cv2.putText(res, "--UNKNOWN--", (720, 400), font, scale, color, thickness)
    show_wrap('WINNER', res, 1080, close=True, video=False)


if __name__ == "__main__":
    """
    Main function:
    1) read the video (change curr_vid for analysis of different video)
    2) prepare variables, tennis court mask (hardcoded) and background subtraction
    3) main 'while' loop for reading the video (new loop = new frame; till there is nothing to read)
        - apply court mask
        - apply background substraction
        - skip ('continue') first few frames for 'BackgroundSubtractorKNN()' to start working
        - if (first loop after initial skipping): initialize meanshift parameters for player tracking
        - else: calculate meanshift
        - track the ball with information about the player bounding boxes
        - print 'hit score'
        - show current frame with all the information
    4) show final frame with the result
    [Press esc during main while loop to end the loop]
    """
    vid_1 = cv2.VideoCapture('../videos/video_cut.mp4')
    vid_2 = cv2.VideoCapture('../videos/video_input8.mp4')
    curr_vid = vid_2

    bag_seg = cv2.createBackgroundSubtractorKNN()
    court_mask = np.zeros((1080, 1920), dtype='uint8')
    result = court_mask
    points = np.array([[140, 1080], [590, 85], [1330, 85], [1780, 1080]])
    cv2.fillPoly(court_mask, pts=np.int32([points]), color=255)
    k, count, switch = 0, 0, 0
    start = 5
    prev_pos = (0, 0)
    entry_pos = (0, 0)
    score = [0, 0]
    near_1, near_2 = False, False
    while True:
        ret, frame = curr_vid.read()
        if not ret:
            break

        result = frame.copy()
        mask_result = frame.copy()
        court = cv2.bitwise_and(mask_result, mask_result, mask=court_mask)
        foreground = bag_seg.apply(court)
        count += 1
        if count < start:
            continue
        morp_forg = morphology(foreground)
        mask_result = apply_mask(morp_forg, mask_result)
        contours, _ = cv2.findContours(morp_forg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)

        if count == start:
            track_window_1, term_crit_1, roi_hist_1 = m_s_setup(mask_result, cnts[0])
            track_window_2, term_crit_2, roi_hist_2 = m_s_setup(mask_result, cnts[1])

        else:
            hsv = cv2.cvtColor(mask_result, cv2.COLOR_BGR2HSV)
            dst_1 = cv2.calcBackProject([hsv], [0], roi_hist_1, [0, 180], 1)
            dst_2 = cv2.calcBackProject([hsv], [0], roi_hist_2, [0, 180], 1)

            _, track_window_1 = cv2.meanShift(dst_1, track_window_1, term_crit_1)
            _, track_window_2 = cv2.meanShift(dst_2, track_window_2, term_crit_2)

            x_1, y_1, w_1, h_1 = track_window_1
            result = cv2.rectangle(result, (x_1, y_1), (x_1 + w_1, y_1 + h_1), 255, 2)
            x_2, y_2, w_2, h_2 = track_window_2
            result = cv2.rectangle(result, (x_2, y_2), (x_2 + w_2, y_2 + h_2), 255, 2)

        result, prev_pos, near_1, near_2, entry_pos = ball_tracking(result, cnts[2:], prev_pos, track_window_1,
                                                                    track_window_2, near_1, near_2, entry_pos)
        result, switch, score = show_hits(result, near_1, near_2, switch, score)
        k = show_wrap('image', result, 1080, close=False, video=True)

        if k == 27:
            curr_vid.release()
            cv2.destroyAllWindows()
            break
    show_win(result, switch)

