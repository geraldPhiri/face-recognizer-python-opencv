import numpy
import cv2
camera=cv2.VideoCapture(0)
wasCaptured,frame=camera.read()                   #wasCaptured will allow to know if we successfully got a frame. it holds True/False
classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
l,n=[],[]
key=ord("a")
while(wasCaptured and (key!=ord("s"))):           #keep taking photos(frames) until s in presses
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   #change img-color to grayscale
    faces=classifier.detectMultiScale(gray,1.3,5) #retrieve faces
    for x,y,w,h in faces:
        resized=cv2.resize(frame[y:y+h,x:x+w],(256,256))
        cv2.imshow("faces",resized)
        key=cv2.waitKey(0)                        #if s is pressed end this collecting of photos process
        if(key=="s"):
            break
        elif(key==ord("g")):                      #if g was pressed associate photo taken with 1
            copy=cv2.cvtColor(resized.copy(),cv2.COLOR_BGR2GRAY)
            l.append(copy)
            n.append(1)                         
        elif(key==ord("d")):                      #if d was pressed associate photo taken with 2
            copy=cv2.cvtColor(resized.copy(),cv2.COLOR_BGR2GRAY)
            l.append(copy)
            n.append(2)                           
    wasCaptured, frame=camera.read()
cv2.destroyAllWindows()

#recognition section
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.train(numpy.asarray(l),numpy.asarray(n))
#recogniser.save("customf_face_recognizer.xml")

wasCaptured,frame=camera.read()
while(wasCaptured):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=classifier.detectMultiScale(gray,1.3,5)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        resized=cv2.resize(gray[y:y+h,x:x+w],(256,256))
        p=recognizer.predict(resized)
        if(p[0]==1):                            #for the photo associated with key=ord("g")
            cv2.putText(frame,"gerald",(x,y-15),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)            
        elif(p[0]==2):                          #for the photo associated with key=ord("d")
            cv2.putText(frame,"daniel",(x,y-15),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2) 
    cv2.imshow("frame",frame)
    cv2.waitKey(2)
    wasCaptured,frame=camera.read()

