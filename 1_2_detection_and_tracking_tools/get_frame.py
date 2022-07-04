import cv2

vcap = cv2.VideoCapture('8.mp4')

vcap.set(1, 15)
x, _ = vcap.read()
print(x)
cv2.imwrite('frame.png', vcap.read()[1]) # extract spcific frame from a video
vcap.close()