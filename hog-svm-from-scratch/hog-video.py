import cv2
import time
import imutils

videofile = 'people.avi'
cap = cv2.VideoCapture(videofile) 

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60, (frame_width,frame_height))

#cap = cv2.VideoCapture(0)  #for webcam

assert cap.isOpened(), 'Cannot capture source'

frames = 0

start = time.time()

cnt = 0

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


while cap.isOpened():
    ret, frame = cap.read()
    
    cnt = 0
    
    if ret:
        
        #frame = imutils.resize(frame,width=min(400,frame.shape[1]))
        (rects,weights) = hog.detectMultiScale(frame,winStride=(4,4),padding=(8,8),scale=1.05)
        for(x,y,w,h) in rects:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 0, 255),2)
        out.write(frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1
        cv2.imshow("frame", frame)
        print(time.time() - start)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
    else:
        break     


cap.release()
out.release()
cv2.destroyAllWindows() 