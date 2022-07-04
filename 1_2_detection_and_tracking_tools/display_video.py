import cv2
import os
import json

FILE_NAME = 'movie_name.txt'

# if os.path.exists(FILE_NAME):
# 	with open(FILE_NAME,'r') as f:
# 		name = f.readline()

with open('app.config') as data:
    config = json.load(data)
count = 1
WIDTH, HEIGHT = 4, 3

vidcap = cv2.VideoCapture(config["input_video"])
height, width = int(vidcap.get(WIDTH)), int(vidcap.get(HEIGHT))
while(vidcap.isOpened()):
	ret, image_np = vidcap.read()
	cv2.imshow('object_detection', cv2.resize(image_np, (width, height)))



	cv2.imwrite(os.path.join('frames', 
                '{}.jpg'.format(str(count).zfill(10))), 
                image_np)
	count += 1

	# this is optional to quit in the middel
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()
    