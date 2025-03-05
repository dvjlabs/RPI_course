# OpenCV

Liberamente ispirandomi a Wikipedia, posso dirvi che OpenCV (acronimo in
lingua inglese di Open Source Computer Vision Library) è una libreria
software multipiattaforma nell'ambito della visione artificiale in
tempo reale.

È una libreria software libera originariamente sviluppato da Intel e poi
presa in carico dalla comunità opensource. Il linguaggio di
programmazione principalmente utilizzato per sviluppare con questa
libreria è il C++, ma è possibile interfacciarsi ad essa anche
attraverso il C, il Python e Java.


## Installazione su RPI3 (Raspbian Buster)


``` py title="Registrazione Video"
pip install opencv-python
pip install picamera
pip install pillow
``` 


## DOCUMENTAZIONE PICAMERA

<a href="https://picamera.readthedocs.io/en/release-1.13/index.html">LINK</a>


## CONTROLLO 

``` py title="Controllo versioni Python e OpenCV"
import sys
print("Versione Python", sys.version)

import cv2
print ("versione OpenCv", cv2.__version__)
```


## IMAGE SHOW

``` py title="Mostrare una immagine con OpenCV"
import cv2

image = cv2.imread('andreagiorgio.webp')
dim = image.shape
print("Dim:", dim)
cv2.imshow("Noi... da giovani!", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```


## VIDEO CAPTURE

``` py title="Registrazione Video"
import cv2

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cam.read()

    # Write the frame to the output file
    out.write(frame)

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()
```

Poi, finalmente...

``` bash
$ sudo pip3 install opencv-contrib-python==3.4.3.18
```


## SMILE DETECTION


``` py title="rettangolo rosso intorno alla bocca che sorride!"
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml') 


def detect(gray, frame): 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    for (x, y, w, h) in faces: 
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2) 
        roi_gray = gray[y:y + h, x:x + w] 
        roi_color = frame[y:y + h, x:x + w] 
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20) 
  
        for (sx, sy, sw, sh) in smiles: 
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2) 
    return frame 


video_capture = cv2.VideoCapture(0) 
while video_capture.isOpened(): 
# Captures video_capture frame by frame 
    _, frame = video_capture.read() 

    # To capture image in monochrome
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # calls the detect() function 
    canvas = detect(gray, frame) 

    # Displays the result on camera feed
    cv2.imshow('Video', canvas) 

    # The control breaks once q key is pressed 
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Release the capture once all the processing is done. 
video_capture.release()
cv2.destroyAllWindows() 
``` 




## LAB 06 - Visi: riconosciamoli

``` bash
import cv2
import sys

#cascPath = sys.argv[1]
#faceCascade = cv2.CascadeClassifier(cascPath)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
```


## LAB 07 - Ogni viso ha i suoi occhi

``` bash
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: #esc 27 ascii
        break

cap.release()
cv2.destroyAllWindows()
```


## LAB 08 - Alla ricerca del colore

Ponete un oggetto di colore verde davanti alla vostra webcam e vediamo
cosa succede. Una pallina sarebbe l'oggetto perfetto.

``` bash
import cv2
import numpy as np
import math

# creo l'oggetto per l'acquisizione del video inserendo 0 il video verra acquisito dalla telcamera,
# inserendo il nome di un file video (posto nella directory del programma) verra aperto quello.
cap = cv2.VideoCapture(0)

# Controllo che la telecamera sia disponibile
if (cap.isOpened()== False):
  print("IMPOSSIBILE ACQUISIRE IL VIDEO!")

# Eseguo finche il video e disponibile

while(cap.isOpened()):
  # leggo frame per frame
  ret, frame = cap.read()
  if ret == True:

    #converto in formato hsv
    frame1= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #applico una sfumatuta per ridurre i disturbi
    frame2= cv2.GaussianBlur(frame1,(5,5),0)

    #definisco i margini di colore da filtrare
    chiaro=np.array([50,80,80])
    scuro=np.array([80,200,200])

    #filtro l'immagine secondo i colori definiti
    frame3=cv2.inRange(frame2, chiaro,scuro)

    #applico una sfumatuta per ridurre i disturbi
    _,frame4= cv2.threshold(frame3,127,255,0)

    #calcolo l'area della superficie colorata e ottengo di conseguenza il centro
    moments = cv2.moments(frame4)
    area = moments['m00']

    radius =int( (math.sqrt((area/3.14)))/10)
    centroid_x, centroid_y = None, None

    if area != 0:
        c_x = int(moments['m10']/area)
        c_y = int(moments['m01']/area)
        print("x: ", c_x,"y: ",c_y)

    # se e stato trovato un oggetto corrispondente ai citeri di ricerca(colore),
    # utilizzo le sue coordinate sullo schermo per tracciare un cerchio attorno ad esso

    if c_x != None and c_y != None:

        # disegno il cerchio
        cv2.circle(frame, (c_x,c_y), radius, (0,0,255),2)

    #disegno la griglia sullo schermo
    cv2.line(frame,(320,0),(320,480),[255,0,0],1)
    cv2.line(frame,(0,240),(640,240),[255,0,0],1)
    cv2.rectangle(frame,(310,230),(330,250),[0,0,255],1)


    # proietto il video acquisito in una finestra
    cv2.imshow('Frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  # esco dal loop
  else:
    break

# chiudo il file video o lo stream della telecamera
cap.release()

# Chiudo la finestra creata
cv2.destroyAllWindows()
```
