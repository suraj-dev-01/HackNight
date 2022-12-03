import os
import cv2
import numpy as np
import face_recognition as fr
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk

path = "./train/"

known_names = []
known_name_encodings = []

images = os.listdir(path)
for _ in images:
    image = fr.load_image_file(path + _)
    image_path = path + _
    encoding = fr.face_encodings(image)[0]

    known_name_encodings.append(encoding)
    known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())

my_w = tk.Tk()
my_w.geometry("700x500")
my_w.title('Face Recognition')
my_font1=('times', 18, 'bold')
l1 = tk.Label(my_w,text='Choose the image',width=30,font=my_font1)  
l1.grid(row=1,column=1)
b1 = tk.Button(my_w, text='Upload File', 
   width=20,command = lambda:upload_file())
b1.grid(row=2,column=1)
global name_label
name_label=Label(my_w, text="", font=('Aerial 14'))
name_label.grid(row=2,column=2)

def compare(test_image):
    image = cv2.imread(test_image)

    face_locations = fr.face_locations(image)
    face_encodings = fr.face_encodings(image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_name_encodings, face_encoding)
        name = ""

        face_distances = fr.face_distance(known_name_encodings, face_encoding)
        best_match = np.argmin(face_distances)

        if matches[best_match]:
            name = known_names[best_match]
            name_label.config(text = name)
        else:
            name = "Unknown Face"
            name_label.config(text = name)

def upload_file():
    global img
    f_types = [('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img=Image.open(filename)
    img_resized=img.resize((400,400))
    img=ImageTk.PhotoImage(img_resized)
    b2 =tk.Button(my_w,image=img)
    b2.grid(row=3,column=1)
    compare(filename)

my_w.mainloop()  # Keep the window open
