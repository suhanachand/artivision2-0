import cv2
#from picamera2 import Picamera2
import struct
import pyaudio
#import pvporcupine
import os, sys
import time
import pytesseract
import pyttsx3
from gtts import gTTS
import pandas as pd
import numpy as np 

from picamera2 import Picamera2

D_ON=1
PICAM=0
LOG=1

#this helps with print statements on console
def logging(text):
  if LOG == 1:
     print(text)

def get_cam_prop(cap):
    print("brightness = ", cap.get(cv2.CAP_PROP_BRIGHTNESS))  # Brightness of the image (only for cameras).
    print("contrast =", cap.get(cv2.CAP_PROP_CONTRAST))  # Contrast of the image (only for cameras).
    print("saturation =", cap.get(cv2.CAP_PROP_SATURATION))  # Saturation of the image (only for cameras)
    print("hue =", cap.get(cv2.CAP_PROP_HUE))  # Hue of the image (only for cameras).
    #white_balance_temperature_auto
    #gamma
    print("gain =", cap.get(cv2.CAP_PROP_GAIN))  # Gain of the image (only for cameras).
    #power_line_frequency
    #white_balance_temperature
    #sharpness
    #backlight_compensation
    #exposure_auto
    #exposure_absolute
    #exposure_auto_priority
    #focus_absolute 1,1023,value=384
    #print("Focus absolute =",cap.get(cv2.CAP_FOCUS_ABSOLUTE))
    #focus_auto,def=1, value=1
    #privacy
    return

def set_cam_prop(cap):
    cap.set(cv2.CAP_PROP_BRIGHTNESS,30)
    print("New brightness = ",cap.get(cv2.CAP_PROP_BRIGHTNESS))
    get_cam_prop(cap)
    return

def speak_text(text):
    # Initialize the engine
    print("speak")
    print(text)
    if text =="":
       text="No text to read"
    engine = pyttsx3.init()
    rate1 = engine.getProperty('rate')
    engine.setProperty('rate', 130)
    rate2 = engine.getProperty('rate')
    #print("Rate: ", rate1," changed to ", rate2)
    engine.say(text) 
    engine.runAndWait()
    return

def alphabets(element):
    str1 = "".join(filter(str.isalpha, element))
    str2 = str1.strip()
    return str2

def get_text(image):
    #boxes = pytesseract.image_to_boxes(image)
    df = pytesseract.image_to_data(image, output_type='data.frame')
    #print(df)
    df.dropna(inplace=True)
    print("###############################")
    print("    After dropping NAN")
    #print(df)
    #print(boxes)
    
    df = df[df.conf > 0] #remove all text which is of conf < 0
    print("    removed low conf")
    #print(df)

    print("    removed non alphanumeric chars")
    df.loc[:,'text'] = [alphabets(x) for x in df.text]
    
    print("    add a new col size")
    size_list = []
    for x in range(len(df)):
        size_list.append("")
    df.loc[:,'size'] = size_list
    df.loc[:,'size'] = [len(x) for x in df.text]
    #print(df)

    print(type(df.size))
    print("     remove all rows with size 0")
    df = df[df['size'] > 0]
    #print(df.loc[:,'text'])
    df.loc[:,'text']=[x.strip() for x in df.text]

    #df = df.strings.str.replace('[^a-zA-Z]', '')
    print(df)
   
    '''
    for i in range(n_boxes)
        if (df['text'][i] != ""):
           (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
           cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) # in-place operation
    return image
    '''
    if (len(df.index))> 0:
       text=""
       num_of_blocks = df['block_num'].max()
       print("##################################")
       print(f'### num of blocks:{num_of_blocks}')
       cropped_flag=0

       '''
       #check if the image is truncated
       for index, row in df.iterrows():
           #if (row["left"] > 0) and (row["left"]+row["width"] < 1279):
           if (row["left"] > 0):
              cropped_flag=0
           else:
              if (row["text"].isspace()):
                 cropped_flag=0
              else:
                 cropped_flag=1
                 break

       print("cropped flag=:", cropped_flag)
       '''
       #form text if not truncated
       if (cropped_flag == 0):
          i=1
          while i <= num_of_blocks:
             df1 = df[(df["block_num"]==i)]
             if (D_ON==1):
                pass
                #print("Block:",i)
                #print(df1)

             for w in df1["text"]:
                 if str(w)=="":
                    pass
                 else:
                    text=text+str(w)+" "
             #text=text+"\n"
             i=i+1
       else:
          print("Can not read text. Please readjust the camera")

       
       #print each levels
       for x in df["line_num"]:
         print(f'{x} : {df[df["line_num"]==x]["text"]}')
 
    else:
       print("    df is empty")
       text=""
       cropped_flag=0

    #return text ,cropped_flag
    return text

def read_image():
    logging("inside read image")
    counter = 0
    prev_word = ""
    '''
    cam1 = cv2.VideoCapture(0)
    cam1.set(3, 640) # set video widht
    cam1.set(4, 480) # set video height
    '''
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format":'XRGB8888', "size": (1640, 1480)}))

    picam2.start()

    #get_cam_prop(cam)
    #set_cam_prop(cam)
    counter = 0

    while True and counter<1:
       print("Cam is opened")
       #ret1, frame1 = cam1.read()
       #cv2.waitKey(5000) 
       img = picam2.capture_array() #8bit image
       #img = (img * 256).astype('uint16') #convert to 16bit
       img = cv2.rotate(img,cv2.ROTATE_180)
       #logging(img.shape, LOG_FLAG)
       #logging(len(cv2.split(img)), LOG_FLAG)
       b, g, r, c = cv2.split(img)
       img = cv2.merge((b,g,r))
       #logging(img.shape, LOG_FLAG)

       cv2.imshow('frame',img)
       cv2.waitKey(15)

       frame1 =  img
       gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
       lapval = cv2.Laplacian(gray, cv2.CV_64F).var()
       print("lap:",lapval)
      
       cv2.imwrite("ImageViewer.jpg", frame1)
       frame2 = cv2.imread("ImageViewer.jpg")
       text = get_text(frame2)
       print(text)
       if text:
          speak_text(text)
          break 
       '''
       ret1 = True
       if ret1 == True:
          #check if image is blurry
          #frame1 = cv2.imread("ImageViewer.jpg")
          frame1 =  img
          gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
          lapval = cv2.Laplacian(gray, cv2.CV_64F).var()
          print("lap:",lapval)
          if lapval < 1850:
             print("blurrry")

          print("ret:", ret1)
          if counter < 1:
              text = get_text(frame1)
              counter = counter+1
          else:
              counter = 0
              print(text)
              cv2.waitKey(0)

          cv2.imshow('frame',frame1)
          cv2.waitKey(1)
       '''
       k = cv2.waitKey(5) & 0xff # Press 'ESC' for exiting video
       if k == 27:
          break# Do a bit of cleanup
    cv2.waitKey(5)
    cv2.destroyAllWindows()
    print("\n [INFO] Exiting Program and cleanup stuff")
    #logging(f'Releasing read cam {cam}')
    #cam.release()
    picam2.stop()
    #cv2.destroyAllWindows()

    return

if __name__ == '__main__':
   print("Hello")
   read_image()
