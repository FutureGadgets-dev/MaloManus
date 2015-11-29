import numpy as np
import cv2
import argparse

totalList=[]

cap = cv2.VideoCapture(0)
nxt = True

finalscore=0
score=0
total=0

while (nxt):
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        vis = np.zeros((480, 640*2), np.float32)
        
        # Our operations on the frame come here
        backGray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Display the resulting frame
        cv2.rectangle(backGray, (180-250,100-250), (460+250,380+250), (0,0,0), 500, 500, 0)
        cv2.putText(backGray, "Welcome User", (235,65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255))
        cv2.putText(backGray, "Your Average is "+str(round(finalscore,2)), (195,85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255))
        cv2.putText(backGray, "Attempt to center hand", (180,405), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255))
        cv2.imshow('frame',backGray)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('x'):
            print ("Captured!")
            cv2.imwrite("Hand.jpg",backGray)
            break

    # When everything done, release the capture
    img = cv2.imread('Hand.jpg')#Hand
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([115,200,100])
    upper_green = np.array([185,260,160])

    mask = cv2.inRange(img, lower_green, upper_green)

    cv2.imshow('Malomanus Readout', mask)
    cv2.imshow('frame',img)

    cv2.imwrite("Temp.jpg",mask)
    tmp=cv2.imread("Temp.jpg")
    final=cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
                
    # Bitwise-AND mask and original image
    ##res = cv2.bitwise_and(frame,frame, mask = mask)

    ##imgray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(final,127,255,0)
    _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    area = 0 #Stores area
    for i in range(0,len(contours)-1): #Checks area of contours
        #if cv2.contourArea(contours[i]) > 10:
        area+=cv2.contourArea(contours[i])
        #Adds area to total if greater than certain amount

    height, width, channels = tmp.shape
    screenArea = 460*380#height*width #Screen res
    percentDirty = ((area/screenArea)*100) #Percentage dirty lazy way
    if percentDirty<0:
        percentDirty=0

    total+=1
    score+=percentDirty
    finalscore=score/total
    
    cv2.drawContours(tmp,contours,-1,(0,255,0),3) #Draws contours for area to screen
##    cv2.putText(img, "Your Score This Time: " +str(percentDirty), (400,85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0))

    print("%.25f%% dirty" % percentDirty)
    if percentDirty > 5: #If a certain amount of germs
        print("Unhygienic; wash further.")
    else:
        print("Pree clean mang")

    k = cv2.waitKey(5) & 0xFF

    cv2.waitKey(0)
    cv2.destroyAllWindows()

print(sum(totalList)/len(totalList))

