# vim: expandtab:ts=4:sw=4
import argparse


import numpy as np
import cv2
import os
import shutil
import colorsys

from deep_sort.iou_matching import iou
from application_util import visualization


DEFAULT_UPDATE_MS = 20
HEIGHT = 3
WIDTH = 4
FRAMES_NUM = 7
c = 0

def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


def rectangle(x, y, w, h, image, color, thickness=2, label=None):
        """Draw a rectangle.

        Parameters
        ----------
        x : float | int
            Top left corner of the rectangle (x-axis).
        y : float | int
            Top let corner of the rectangle (y-axis).
        w : float | int
            Width of the rectangle.
        h : float | int
            Height of the rectangle.
        label : Optional[str]
            A text label that is placed at the top left corner of the
            rectangle.

        """
        pt1 = int(x), int(y)
        pt2 = int(x + w), int(y + h)
        cv2.rectangle(image, pt1, pt2, color, thickness)
        if label is not None:
            text_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_PLAIN, 1, thickness)

            center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
            pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + \
                text_size[0][1]
            cv2.rectangle(image, pt1, pt2, color, -1)
            cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN,
                        1, (255, 255, 255), thickness)

def draw_groundtruth(track_id, box, img):
    thickness = 2
    color = create_unique_color_uchar(track_id)
    rectangle(*box.astype(np.int), img, color, thickness, label=str(track_id))

def write_video_with_object_i(orig_video_path, object_i, tracking_of_object_i, avg_w, avg_h, fourcc_string='mp4v'):

    def extract(x, y, w, h, img):
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        if (x2 <= x1 or y2 <= y1):
            return None, False
        return img[y1:y2, x1:x2, :], True
        

    vcap = cv2.VideoCapture(orig_video_path)
    fps = vcap.get(cv2.CAP_PROP_FPS)
    img_size = (avg_w, avg_h)
    
    fourcc = cv2.VideoWriter_fourcc(*fourcc_string) # Be sure to use lower case

    objects_path = 'Tracking_Videos'
    if not os.path.exists(objects_path):
        os.makedirs(objects_path)

    vwriter = cv2.VideoWriter('Tracking_Videos/{}.mp4'.format(object_i), 
        fourcc, fps,  img_size)
    frame_idx = 0
    frames_of_object_i = tracking_of_object_i[:, 0]

    while True:
        ret, img = vcap.read()
        if not ret:
            vcap.release()
            vwriter.release()
            break
                
        if frame_idx in frames_of_object_i:
            frame_mask = tracking_of_object_i[:, 0].astype(np.int) == frame_idx
            box = tracking_of_object_i[frame_mask, 2:6][0]
            

            output_frame, ret = extract(*box, img)
            if ret:
                vwriter.write(cv2.resize(output_frame, img_size, interpolation = cv2.INTER_LINEAR))    
            
        frame_idx += 1


def save_video_results(orig_video_path):
    if not os.path.isfile('Tracking_Results/tr.csv'):
        print('There are no tracked objects. Execut run.py first.')
        return 

    # delete videos folder if exists
    if os.path.exists('Tracking_Videos'):
        shutil.rmtree('Tracking_Videos', ignore_errors=True, onerror=None)


    results = np.loadtxt('Tracking_Results/tr.csv', delimiter=',')
    # select all objects unique
    ids = np.unique(results[:, 1].astype(int))

    # find the averge size for each object id
    for i in ids:
        mask_of_object_i = results[:, 1] == i
        tracking_of_object_i = results[mask_of_object_i, :]
        avg_w, avg_h = (int(np.average(tracking_of_object_i[:, 4])), int(np.average(tracking_of_object_i[:, 5])))
        
        write_video_with_object_i(orig_video_path, i, tracking_of_object_i, avg_w, avg_h)

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Saving Results")
    
    parser.add_argument(
        "--source_video", help="Enter source video file name",
        default='sample.mp4', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_video_results(args.source_video)

    

    




    


