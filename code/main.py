import cv2
import numpy as np
import os

def load_videos_from_folder(folder):
    """
    :param folder: [string] - path to folder with videos
    :return: [np.array of np.ndarray, np.array] - array of videos and array of corresponding filenames
    Load from folder
    """
    videos = []
    filenames = []
    for filename in os.listdir(folder):
        vid = cv2.imread(os.path.join(folder, filename))
        if vid is not None:
            videos.append(img)
            filenames.append(filename)
    return videos, filenames

if __name__ == "__main__":
    vid_arr, filenames = load_videos_from_folder('../videos')
