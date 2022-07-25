import numpy as np
import cv2

capture = cv2.VideoCapture(0)

frame_count = 0
while capture.isOpened():
    ret, frame = capture.read()
    if ret:
        frame_count += 1
        cv2.imshow('frame',frame)
        #print(frame.shape)
    else:
        break
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

print( 'frames {0}'.format(frame_count))
