import cv2
from lib.device import Camera
import time

subject = 'who'
q = 0

cap = cv2.VideoCapture(0)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
fps = 25
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
start = False

out = cv2.VideoWriter(subject + '_q' + str(q) + '.mp4', fourcc, fps, sz)
tStart = time.time()

while(cap.isOpened()):
	_, frame = cap.read()

	out.write(frame)

	cv2.imshow('frame',frame)

	key = cv2.waitKey(10)
	if key & 0xFF == ord('q'):
		break
	elif key & 0xFF == ord('s'):
		if start : print("time of question " + str(q) + ": " +str(time.time()-tStart))
		out.release()

		if not start: start = True
		else : q+=1

		tStart = time.time()
		print("\nRecording..." + subject + '_q' + str(q) + '.mp4')
		out = cv2.VideoWriter(subject + '_q' + str(q) + '.mp4', fourcc, fps, sz)

cap.release()
cv2.destroyAllWindows()