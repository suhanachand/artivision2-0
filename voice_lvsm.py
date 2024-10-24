#https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/1
import cv2
import dlib
import os
import pickle
import time
import numpy as np

from picamera2 import Picamera2

#from gtts import gTTS
import speech_recognition as sr
import pyttsx3
#import pytsx3_voices 
import sounddevice #to remove all alsa errors

from fer_det import model_scan 
from utils import face_rects
from utils import face_encodings
from utils import nb_of_matches
from image_reader import read_image 

from code_logging import logging
#used for counting the freq of faces identified
user_count = []

#load the encoding dictionary
with open("encodings.pickle","rb") as f:
   name_encodings_dict = pickle.load(f)


def add_user(name_list):
   print("Inside add_user")
   check = 0
   for i in range(len(user_count)):
       if user_count[i][0] == name_list[0]:
          user_count[i][1] = name_list[1]
          user_count[i][2] +=1
          check=1
   if check == 0:
     #name is not added, so add it
     user_count.append([name_list[0], name_list[1], 1])

def tell_user():
   if len(user_count) > 0:
      print("inside tell user:",user_count)
      user_count=[]

 
# Function to convert text to
# speech
def SpeakText(gen_text):
    # Initialize the engine
    print("inside user count")
    engine = pyttsx3.init()
    rate1 = engine.getProperty('rate')
    engine.setProperty('rate', 130)
    rate2 = engine.getProperty('rate')
    print("Rate: ", rate1," changed to ", rate2)

    print(gen_text)
    engine.say(gen_text) 
    engine.runAndWait()

def test_rec_face():
   img = cv2.imread("test.jpeg")
   un_encodings = face_encodings(img)
   if len(un_encodings) == 0:
      print("No encodings found")
   names = []      
   for un_encoding in un_encodings:
      counts = {}
      for (name, encodings) in name_encodings_dict.items():
            counts[name] =  nb_of_matches(encodings, un_encoding)

      if all(count == 0 for count in counts.values()):
         name = "Unknown"
      else:
         name =  max(counts, key=counts.get)
      names.append(name)
   print("Names:", name)
   for rect, name in zip(face_rects(img), names):
       # get the bounding box for each face using the `rect` variable
       x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
       #get feelings
       sub_image = img[y1-5:y2+5,x1-5:x2+5]
       #cv2.imshow("sub image:", sub_image)
       #cv2.waitKey(0)
       #cv2.destroyAllWindows()
       print("Sub_image size:",sub_image.shape())
       feelings = model_scan(sub_image)
       print("Feelings:", feelings)

       text_label = name+": "+feelings
       name_label = [name, feelings]
       # draw the bounding box of the face along with the name of the p>
       cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
       cv2.putText(img, text_label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
       add_user(name_label)
   # show the output image
   cv2.imshow("image", img)
   cv2.waitKey(0)
   print("\n [INFO] Exiting Program and cleanup stuff")
   cv2.destroyAllWindows()

def recognize_face():
   #cam = cv2.VideoCapture(0)
   #cam.set(3, 640) # set video widht
   #cam.set(4, 480) # set video height

   picam2 = Picamera2()
   picam2.configure(picam2.create_preview_configuration(main={"format":'XRGB8888', "size": (1640, 1480)}))

   picam2.start()

   timer_reset_flag=0
   print("Inside Recognize Face.....")
   kount=0

   while True:
      print("Kount: ", kount)

      #ret, img =cam.read()
      #pcie cm camera
      img = picam2.capture_array() #8bit image
      #img = (img * 256).astype('uint16') #convert to 16bit
      img = cv2.flip(img,0)
      #logging(img.shape, LOG_FLAG)
      #logging(len(cv2.split(img)), LOG_FLAG)
      b, g, r, c = cv2.split(img)
      img = cv2.merge((b,g,r))
      #logging(img.shape, LOG_FLAG)
      
      ret=True
      if ret:
         #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
         if timer_reset_flag == 0:
            start_time = time.time()
            timer_reset_flag = 1
            print("Start timer:",start_time)

         unknown_encodings = face_encodings(img)
         if len(unknown_encodings) == 0:
            print("No encodings found")

         names = []

         for unknown_encoding in unknown_encodings:
            counts = {}
            for (name, known_encodings) in name_encodings_dict.items():
                counts[name] =  nb_of_matches(unknown_encoding, known_encodings)

            if all(count == 0 for count in counts.values()):
               name = "Unknown"
            else:
               name =  max(counts, key=counts.get)

            names.append(name)

         print("Names:", names)

         for rect, name in zip(face_rects(img), names):
             # get the bounding box for each face using the `rect` variable
             x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()

             #get feelings
             sub_image = img[y1-5:y2+5,x1-5:x2+5]
             #cv2.imshow("sub image:", sub_image)
             #cv2.waitKey(0)
             #cv2.destroyAllWindows()
             feelings = model_scan(sub_image)
             print("Feelings:", feelings)

             text_label = name+": "+feelings
             name_label = [name, feelings]
             # draw the bounding box of the face along with the name of the p>
             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
             cv2.putText(img, text_label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

             '''
             print("current time:", time.time())
             if ((time.time() - start_time) > 5):
                #Its time to tell the user
                break
             else:
                add_user(name_label)
             '''
             add_user(name_label)

         # show the output image
         cv2.imshow("image", img)
         k = cv2.waitKey(700) & 0xff # Press 'ESC' for exiting video
         if k == 27:
            break# Do a bit of cleanup
         kount=kount+1
         if kount == 4:
            cv2.destroyAllWindows()
            break

   print("\n [INFO] Exiting Program and cleanup stuff")
   #print(f'Releasing show cam {cam}')
   #cam.release()
   #cv2.destroyAllWindows()
   if len(user_count) > 0:
      print("names:", user_count)
      for i in range(0,len(user_count)):
          #print("Faces identified:",user_count[i][0])
          gen_text = "I found "+ user_count[i][0]+ " who has a "+user_count[i][1]+" face"
          print(gen_text)
          SpeakText(gen_text)

   picam2.close()

   return

if __name__ == "__main__" :
    # Initialize the recognizer 
    r = sr.Recognizer() 
    print("r:", r)

    '''
    SpeakText("Hi,welcome to Artivision2.O")
    SpeakText("Please choose from the 2 options:")
    SpeakText("Option1: Read Text")
    SpeakText("Option2: Show Faces")
    SpeakText("If you want to exit say quit")
    '''

    no_answer_counter = 0

    while(1):
       try:
          # use the microphone as source for input.
          with sr.Microphone() as source2:
             print("Speak ----")

             no_answer_counter=+1

             if (no_answer_counter < 5):
                # wait for a second to let the recognizer
                # adjust the energy threshold based on
                # the surrounding noise level 
                r.adjust_for_ambient_noise(source2, duration=0.2)

                #listens for the user's input 
                audio2 = r.listen(source2)
                print(audio2)

                # Using google to recognize audio
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()

                print("Did you say ",MyText)
                if (MyText == "show faces" or MyText == "option 2" or MyText == "option two"):
                   print("Voice command recognized")
                   recognize_face()
                   #time.sleep(1)
                   #print("Voices:", engine.getProperty('voices'))
                   '''
                   #speak to the user and tell him about the faces recognized
                   if len(user_count) > 0:
                      for i in range(0,len(user_count)):
                         #print("Faces identified:",user_count[i][0])
                         gen_text = "I found "+ user_count[i][0]+ " who has a "+user_count[i][1]+" face"
                         print(gen_text)
                         SpeakText(gen_text)
                   '''
                elif MyText == "read text" or MyText == "option 1" or MyText == "option one":
                   read_image()

                elif MyText == "quit":
                   SpeakText("Thank you for using Artivision2.0")
                   break

                else:
                   print("Invalid command. Re say your command")
                   SpeakText("Invalid Entry. Please say your command.")
             else:
                SpeakText("I havent heard a choice from you. Please choose between Show Faces and read text")

       except sr.RequestError as e:
          print("Could not request results; {0}".format(e))

       except sr.UnknownValueError:
          print("unknown error occurred")
