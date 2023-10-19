import cv2
from Model import Model
import PIL.Image 
from emothion_graph import CloseToDots
import numpy as np

class EmotionDetector:
    def __init__(self) :
        # prepare my model
        self.my_model = Model() 

        names = ["angry","disgust","fear","happy","neutral","sad","surprise"]
        poses = [[2.5,2],[1,2],[1,3],[4,3],[2.5,3],[2,4],[3,4]]

        self.dots= CloseToDots(poses,names)

        smooth = 5
        self.yhatList = np.zeros(shape=(smooth,7))
        

        # Load the pre-trained Haar Cascade classifier for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def openCamera(self):
        # Open a video stream using the webcam (change the argument to a video file path if needed)
        video_capture = cv2.VideoCapture(0)
        self.detect(video_capture)
    
    def openVideo(self,path):
        # Open a video stream using the webcam (change the argument to a video file path if needed)
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        if(ret == False):
            raise Exception("invalid path")

        self.detect(cap)

    def detectOnFrame(self,frame):
        gray_frame=frame
        try:gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except :None
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)


        for i,(x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            x,y,w,h = faces[i]
            x2 = x+w
            y2=y+h
            face = gray_frame[y:y2,x:x2]
            emotion_text,yhat = self.my_model.predict(PIL.Image.fromarray(face))

            self.yhatList=np.concatenate([self.yhatList[1:],yhat.detach().numpy()])

            textScale = 1 if w<50 else 2
            cv2.putText(frame,emotion_text,(x + w, y + h),1,textScale,(0, 255, 0))

            # syhat= np.average(self.yhatList,axis=0)
        
            # self.dots.show_dots(syhat)
            
        return frame


    def detect(self,video_capture:cv2.VideoCapture,detectEveyNframes=0):

        passedFrames = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break  # Break the loop if the video ends
            if passedFrames>detectEveyNframes:
                passedFrames=0
                frame = self.detectOnFrame(frame)
            cv2.imshow('Video', frame)
            passedFrames+=1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Press 'q' to exit the loop

        video_capture.release()
        cv2.destroyAllWindows()
        self.dots.show()


if __name__ =="__main__":
    em = EmotionDetector()
    em.openCamera()
    