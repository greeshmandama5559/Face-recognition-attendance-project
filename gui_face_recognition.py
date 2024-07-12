import cv2
import numpy as np
import face_recognition
import os
import csv
from datetime import datetime, date
import winsound
from tkinter import *
import tkinter as tk
from tkinter import messagebox
import pickle
from PIL import Image,ImageTk
from tkinter import simpledialog

def encodings():
    known_face_encoding = []
    known_faces_names = []
    img = []
    image_names = os.listdir(folder_path)
    for f in image_names:
        curimg = cv2.imread(f'{folder_path}/{f}')
        img.append(curimg)
        known_faces_names.append(os.path.splitext(f)[0])

    for img1 in img:
        if img1 is not None:
            img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        img_enco = face_recognition.face_encodings(img1)[0] 
        if len(img_enco)>0:
            known_face_encoding.append(img_enco)
        else:
            print("somthing went wrong, the image in folder may not contain face!!")

    data_to_save = (known_face_encoding, known_faces_names)
    with open(encodings_file, 'wb') as f:
        pickle.dump(data_to_save, f)

    messagebox.showinfo("Success","Faces are successfully encoded!")


def open_capture_window():
    cap = cv2.VideoCapture(0)
    x, y = 40, 78
    while True:
        ret, frame = cap.read()
        if not ret:
            print("cannot capture")
        frame = cv2.resize(frame, (475, 362),None,0.20,0.20)
        imgBG[y:y+362, x:x+475] = frame
        cv2.imshow("frame",imgBG)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            image_name = simpledialog.askstring("Input", "Enter Student RollNo.:  ")
            if image_name:
                image_path = os.path.join(destination_folder, f"{image_name}.jpg")
                cv2.imwrite(image_path, frame)
                print(f"{image_name}.jpg is saved")
        if key == ord('e'):
            break
    cap.release()       
    cv2.destroyAllWindows()

def load_saved_encodings(encodings_file):
    if os.path.exists(encodings_file):
        with open(encodings_file, 'rb') as f:
            known_face_encoding, known_faces_names = pickle.load(f)
            return known_face_encoding, known_faces_names
    else:
        print("No encodings found")

def run_face_recognition():
    global students
    global video

    frequency = 1000
    duration = 600
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video = cv2.VideoCapture(0)

    encodings_file = 'face_encodings/face_encodings.pkl'
    known_face_encoding, known_faces_names = load_saved_encodings(encodings_file)

    students = known_faces_names.copy()
    face_locations = []
    face_encodings = []

    now = datetime.now()
    cur_date = now.strftime("%Y-%m-%d")
    name = ""
    names = []

    f = open(cur_date+'.csv',"w+",newline="")
    imwriter = csv.writer(f)
    imwriter.writerow(["roll_number","Time"])

    video.set(cv2.CAP_PROP_FPS, 40)

    while True:
        _, frame = video.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(rgb_frame, (0, 0), None, 0.50, 0.50)
        faces = face_cascade.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))

        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        cv2.imshow("Attendance", recog_bg)
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding, tolerance=0.4)
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Rollno:{name}", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            else:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        frame = cv2.resize(frame,(460,371),None,0.50,0.50)
        x,y = 171,54
        recog_bg[y:y+371,x:x+460] = frame

        if name in students:
            students.remove(name)
            cur_time = now.strftime("%H:%M")
            today = date.today()
            d2 = today.strftime("%B %d, %Y")
            print(name, " ", cur_time, " ", d2)
            names.append(name)
            winsound.Beep(frequency, duration)

        if cv2.waitKey(1) & 0xFF == ord('e'):
            break
    names.sort()
    for roll in names:
        imwriter.writerow([roll,cur_time])
    video.release()
    cv2.destroyAllWindows()
    f.close()

root = tk.Tk()
root.geometry('700x410')
root.title("Face Recognition")
canvas = Canvas(root,bg="#5D64A1",height="410",width="700",relief="flat")
canvas.pack()
canvas.place(x = -2, y = 0)
bg = ImageTk.PhotoImage(Image.open("face_bg.png"))
canvas.create_image(352,205,image=bg)

run_button = tk.Button(root, text="Run Face Recognition",bg="cyan", fg="black",command=run_face_recognition,borderwidth=3)
run_button.place(x=290,y=330)

register_button = tk.Button(root, text="Add New Student", bg="steelblue",fg="black",command=open_capture_window,borderwidth=3)
register_button.place(x=355, y=365)

encode_button = tk.Button(root, text="Encode faces",bg="steelblue", fg="black",command=encodings,borderwidth=3)
encode_button.place(x=265,y=365)

folder_path = "photos" 
encodings_file = 'face_encodings/face_encodings.pkl'
recog_bg = cv2.imread("recognition_bg (1).png")
imgBG = cv2.imread("imgBG1.png")
destination_folder = "photos"
root.mainloop()
