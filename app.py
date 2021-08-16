import cv2
import numpy as np
import streamlit as st
import pytesseract
from PIL import Image
st.title('Car Number plate detection and recognition')
img = st.sidebar.file_uploader('Choose an image')
if img is not None:
  img_read = Image.open(img)
  img_new = np.array(img_read)
  st.image(img_new,caption='Image Uploaded')
  
  if st.button('EXTRACT'):
    plate = cv2.CascadeClassifier('/content/haarcascade_russian_plate_number.xml')
    demo_plate = plate.detectMultiScale(img_new,1.1,10)
    #st.image(img_new)  

    for (x,y,w,h) in demo_plate:
      cv2.rectangle(img_new,(x,y),(x+w,y+h),(0,0,255),3)
      
     #Extracting the number plate from image 
      roi = img_new[y:y+h,x:x+w]
      st.title("Extracted Image:")
      st.image(roi)

     #Extracting the characters from number plate
      st.title("Extracted by OCR:")
      op = pytesseract.image_to_string(roi)
      st.write(op)
