#!/usr/local/bin/python3

import cv2
import argparse
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-pth", "--path", required=False, default='imgs2', help="images folder path.")
ap.add_argument("-ext", "--extension", required=False, default='jpg', help="extension name. default is 'jpg'.")
ap.add_argument("-o", "--output", required=False, default='videos/output.mp4', help="output video file")
args = vars(ap.parse_args())

# Arguments
dir_path = 'figs'
ext = 'png'
output = 'y.mp4'

images = []
for f in os.listdir(dir_path):
    if f.endswith(ext):
        images.append(f)

print(images)
# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
print(image_path)
frame = cv2.imread(image_path)
# cv2.imshow('video',frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 20, (width, height))

for i in range(1, 2590):

    image_path = os.path.join(dir_path, '{}.png'.format(i))
    frame = cv2.imread(image_path)
    # write 24 frames per second
    #for i in range(24):
    out.write(frame) # Write out frame to video

    # cv2.imshow('video',frame)
    # if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
    #     break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))
