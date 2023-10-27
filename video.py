import cv2 # opencv 4.2.0
cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    #cv2.imshow('cap', frame)
    # 另一种写法：if cv2.waitKey(1) & 0xFF == 27:
    # & 0xFF 的意思是取前面语句返回值的后八位。27对应esc键。https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()