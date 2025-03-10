# OpenCV


## Ambiente di lavoro


Vogliamo creare un ambiente virtuale, in modo da ottenere tre risultati importanti:

1. Provare a 'standardizzare' il lavoro da fare per tutti i Sistemi Operativi

2. Partire da una installazione locale di Python 'pulita', ovvero senza alcun modulo installato (nell'ambiente virtuale)

3. un altro motivo

Creiamo una cartella su cui faremo tutte le nostre prove e i nostri esperimenti ed entriamo con il terminale su di essa.

``` bash
$ pwd
/home/adjam
$ mkdir ProveOpenCV
$ cd ProveOpenCV
```

Adesso creiamo un ambiente virtuale e attiviamolo!

``` bash
$ python -m venv venv
$ source venv/bin/activate.sh
```

!!! tip "Suggerimento"
    La creazione (e attivazione) di un ambiente virtuale si può fare anche direttamente dall'interfaccia di Thonny:
    dal menù ESEGUI , scheda INTERPRETE , in basso a dx, seleziona NUOVO AMBIENTE VIRTUALE e crealo nella cartella `venv`.

    Al termine della creazione dell'ambiente virtuale, aprite il terminale tramite Thonny: dal menù STRUMENTI, selezionate APRI SHELL DI SISTEMA...


Per verificare che l'ambiente è effettivamente `pulito`, digitiamo:

``` bash
$ pip list
Package Version
------- -------
pip     24.3.1
```

Dovrebbe essere elencato solo pip o pochissimo altro.



## Installazione OpenCV

Dal terminale ove si è attivato l'ambiente virtuale (o da quello aperto con Thonny), digitiamo:

``` bash
$ pip install numpy
$ pip install pillow
$ pip install opencv-python
```

Installiamo prima la libreria `numpy` perché è una diretta dipendenza di OpenCV e perché potrebbe dare problemi nell'installazione.
Controlliamo bene che sia andato tutto liscio!


!!! note "Nota"
    La libreria `pillow` è la libreria standard *de-facto* per la gestione delle immagini in Python. Pillow è un gioco di parole fra
    *cuscino* (che si mette fra la testa e il letto) e *LOW PIL*, ovvero *Low Python Image Library*, nel senso che tutte le librerie
    Python usano questa per la gestione delle immagini.

Per verificare che tutto è ok, proviamo il nostro primo test!

``` py title="File: test0.py"
# Controllo versioni Python e OpenCV
import sys
print("Versione Python", sys.version)

import cv2
print ("versione OpenCv", cv2.__version__)
```


## Gestione immagini con OpenCV


Proviamo dunque a gestire una immagine con OpenCV! Scaricate una immagine qualunque (per me, il file `pippo.jpg`)
ed eseguite questo codice:

``` py title="File: testImage1.py"
# Mostrare una immagine con OpenCV
import cv2

image = cv2.imread('pippo.jpg')
print("Dimensione:", image.shape)
cv2.imshow("Il nostro amico Pippo!", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Volendo, è possibile mostrare una immagine in scala di grigi:

``` py title="File: testImage2.py"
# Mostrare una immagine (in scala di grigi) con OpenCV
import cv2

image = cv2.imread('pippo.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Il nostro amico Pippo!", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Chiaramente il focus principale della libreria è l'elaborazione video in tempo reale. Per fare questo, è necessario
gestire la videocamera con Python!


## Videocamera


La videocamera dovrebbe essere supportata nativamente da OpenCV, nel senso che se hai una videocamera installata, questi dovrebbe
essere in grado di riconoscerla e utilizzarla!

Ecco un esempio banale per verificarlo subito!!!

``` py title="Cattura video con OpenCV"
import cv2

videoCapture = cv2.VideoCapture(0)

while True:
    # legge 1 frame dalla videocamera...
    ret, frame = videoCapture.read()
    # e lo visualizzo!
    cv2.imshow('Camera',frame)

    # se premi Q, è tutto finito!
    if cv2.waitKey(1) == ord('q'):
        break

# release the camera from video capture
cap.release()
# De-allocate any associated memory usage
cv2.destroyAllWindows()
```

Se funziona tutto... benissimo!!! Puoi andare avanti Al prossimo capitolo!!! Se invece la videocamera non viene riconosciuta, ad esempio
la picamera sul Raspberry, occorre provare qualcosa di diverso!

Per quanto riguarda il **Raspberry Pi** purtroppo siamo in un momento un pò imbarazzante: il vecchio software non funziona più bene, per mancanza
di compatibilità con gli aggiornamenti del resto del sistema, mentre il nuovo non ha ancora raggiunto un livello soddisfacente di stabilità
e funzionalità!

In ogni caso, siamo costretti ad affidarci alla nuova versione del software di gestione della camera! Il vecchio non funziona...

Per installare tutto su un Raspberry Pi, digitate sul terminale:

``` bash
$ sudo apt install python3-camera2
$ sudo apt install rpicam-apps
```

La nuova soluzione è molto moderna: utilizza la nuovissima libreria `libcamera`, che sarà la libreria di supporto alle videocamere di default
su Android, Linux, Raspberry Pi. Purtroppo non è ancora... 100% pronta!!! Speriamo bene...

Documentazione relativa:

- <a href="https://www.raspberrypi.com/documentation/computers/camera_software.html" target="_blank">rpicam-apps</a>
- <a href="https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf" target="_blank">picamera2 python library</a>

Nella documentazione trovate tutti gli esempi visti sotto relativi a picamera2!


!!! warning "Attenzione!"
    Non installate `picamera2` tramite `pip`, ma fate esattamente come spiegato sopra tramite `apt`!

    Vero che su pip potreste trovare una versione pià *nuova* del software, ma forse non completamente compatibile
    con la versione di libcamera che trovate installata sul Raspberry!


## Esempi con OpenCV e videocamera


Negli esempi che seguono andremo a testare alcune interessanti funzionalità della libreria OpenCV.

Per potenziare le sue già eccellenti capacità, la libreria fornisce dei *classificatori*, denominati `haar-cascades`
che sono utilizzati per implementare funzionalità specifiche (ad esempio, il rilevamento dei volti, degli occhi, del sorriso, etc...)

Solitamente le informazioni necessarie per le funzionalità sono fornite in formato XML. Tutti i file haar-cascades sono disponibili per il
download qui: <a href="https://github.com/opencv/opencv/tree/master/data/haarcascades" target="_blank">https://github.com/opencv/opencv/tree/master/data/haarcascades</a>

In ogni esempio viene indicato chiaramente quali sono i file necessari: vanno scaricati e collocati nella cartella del file Python.

---

Nel primo esempio una prova abbastanza banale: la registrazione di un video tramite le funzionalità di OpenCV!


``` py title="Registrazione Video"
import cv2

videoCapture = cv2.VideoCapture(0)
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
outputVideoFile = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = videoCapture.read()
    outputVideoFile.write(frame)
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

videoCapture.release()
outputVideoFile.release()
cv2.destroyAllWindows()
```


Esempio di SMILE DETECTION: un rettangolo rosso appare intorno ad ogni bocca che sorride!


``` py title="Smile Detection"
import cv2

# file necessari (da scaricare e posizionare nella cartella del programma)
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


videoCapture = cv2.VideoCapture(0)
while videoCapture.isOpened():
    ret, frame = videoCapture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)

    if cv2.waitKey(1) == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()
```

La FACE-DETECTION è l'operazione di OpenCV che la rende in grado di rilevare un viso e distinguerlo
da un pallone, da un piatto, da un quadro, etc...


``` py title="Face Detection"
import cv2
import sys

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

videoCapture = cv2.VideoCapture(0)

while True:
    ret, frame = videoCapture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()
```


Altra potenzialità di OpenCV: la possibilità di riconoscere gli occhi di una persona e individuarli all'interno di un viso!


``` py title="Riconoscimento degli occhi"
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

videoCapture = cv2.VideoCapture(0)

while True:
    ret, img = videoCapture.read()
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
    if cv2.waitKey(1) == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()
```


Ponete un oggetto di colore verde davanti alla vostra webcam e vediamo
cosa succede. Una pallina sarebbe l'oggetto perfetto.


``` py title="Riconoscimento Colori"
import cv2
import numpy as np
import math

videoCapture = cv2.VideoCapture(0)

while videoCapture.isOpened():
    ret, frame = videoCapture.read()
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame2 = cv2.GaussianBlur(frame1,(5,5),0)
    chiaro = np.array([50,80,80])
    scuro = np.array([80,200,200])
    frame3 = cv2.inRange(frame2, chiaro,scuro)
    ret2,frame4 = cv2.threshold(frame3,127,255,0)
    moments = cv2.moments(frame4)
    area = moments['m00']

    radius =int( (math.sqrt((area/3.14)))/10)
    centroid_x, centroid_y = None, None

    if area != 0:
        c_x = int(moments['m10']/area)
        c_y = int(moments['m01']/area)
        print("x: ", c_x,"y: ",c_y)

    # se trovato, nelle sue coordinate disegno cerchio e griglia!
    if c_x != None and c_y != None:
        cv2.circle(frame, (c_x,c_y), radius, (0,0,255),2)

    cv2.line(frame,(320,0),(320,480),[255,0,0],1)
    cv2.line(frame,(0,240),(640,240),[255,0,0],1)
    cv2.rectangle(frame,(310,230),(330,250),[0,0,255],1)
    cv2.imshow('Frame',frame)

    if cv2.waitKey(1) == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()
```


## Face Recognition


Il riconoscimento del viso è una operazione successiva al semplice rilevamento: questa funzionalità rende i nostri computer
(ma anche telefoni, tablet, portatili, etc...) dotati di videocamera di riconoscere la persona che hanno dinnanzi, non semplicemente
di individuarne il viso, uguale per tutte le persone (che hanno la testa), ma di distinguerle fra loro in base ad un confronto
con alcune immagini di test da fornire per il database.

Per procedere con l'installazione, dal solito terminale con l'ambiente virtuale attivo, digitate:


``` bash
$ pip install setuptools
$ pip install dlib
$ pip install face-recognition
$ pip install imutils
```

Poi posso procedere così...


``` py title="Fai le foto alle persone da riconoscere"
import cv2

name = 'silvia' #replace with your name

cam = cv2.VideoCapture(0)

cv2.namedWindow("press space to take a photo", cv2.WINDOW_NORMAL)
cv2.resizeWindow("press space to take a photo", 500, 300)

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("press space to take a photo", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "dataset/"+ name +"/image_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()
```

*Allena* il modello per il riconoscimento facciale...

``` py title="Train Model"
# import the necessary packages
from imutils import paths
import face_recognition
#import argparse
import pickle
import cv2
import os

# our images are located in the dataset folder
print("[INFO] start processing faces...")
imagePaths = list(paths.list_images("dataset"))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1,
        len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # load the input image and convert it from RGB (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb,
        model="hog")

    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)

    # loop over the encodings
    for encoding in encodings:
        # add each encoding + name to our set of known names and
        # encodings
        knownEncodings.append(encoding)
        knownNames.append(name)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
```

Infine, vai con il riconoscimento facciale...

 ``` py title="Face Recognition"
 # import the necessary packages
 from imutils.video import VideoStream
 from imutils.video import FPS
 import face_recognition
 import imutils
 import pickle
 import time
 import cv2

 #Initialize 'currentname' to trigger only when a new person is identified.
 currentname = "unknown"
 #Determine faces from encodings.pickle file model created from train_model.py
 encodingsP = "encodings.pickle"

 # load the known faces and embeddings along with OpenCV's Haar
 # cascade for face detection
 print("[INFO] loading encodings + face detector...")
 data = pickle.loads(open(encodingsP, "rb").read())

 # initialize the video stream and allow the camera sensor to warm up
 # Set the ser to the followng
 # src = 0 : for the build in single web cam, could be your laptop webcam
 # src = 2 : I had to set it to 2 inorder to use the USB webcam attached to my laptop
 vs = VideoStream(src=0,framerate=10).start()
 #vs = VideoStream(usePiCamera=True).start()
 time.sleep(2.0)

 # start the FPS counter
 fps = FPS().start()

 # loop over frames from the video file stream
 while True:
     # grab the frame from the threaded video stream and resize it
     # to 500px (to speedup processing)
     frame = vs.read()
     frame = imutils.resize(frame, width=500)
     # Detect the fce boxes
     boxes = face_recognition.face_locations(frame)
     # compute the facial embeddings for each face bounding box
     encodings = face_recognition.face_encodings(frame, boxes)
     names = []

     # loop over the facial embeddings
     for encoding in encodings:
         # attempt to match each face in the input image to our known
         # encodings
         matches = face_recognition.compare_faces(data["encodings"],
             encoding)
         name = "Unknown" #if face is not recognized, then print Unknown

         # check to see if we have found a match
         if True in matches:
             # find the indexes of all matched faces then initialize a
             # dictionary to count the total number of times each face
             # was matched
             matchedIdxs = [i for (i, b) in enumerate(matches) if b]
             counts = {}

             # loop over the matched indexes and maintain a count for
             # each recognized face face
             for i in matchedIdxs:
                 name = data["names"][i]
                 counts[name] = counts.get(name, 0) + 1

             # determine the recognized face with the largest number
             # of votes (note: in the event of an unlikely tie Python
             # will select first entry in the dictionary)
             name = max(counts, key=counts.get)

             #If someone in your dataset is identified, print their name on the screen
             if currentname != name:
                 currentname = name
                 print(currentname)

         # update the list of names
         names.append(name)

     # loop over the recognized faces
     for ((top, right, bottom, left), name) in zip(boxes, names):
         # draw the predicted face name on the image - color is in BGR
         cv2.rectangle(frame, (left, top), (right, bottom),
             (0, 255, 225), 2)
         y = top - 15 if top - 15 > 15 else top + 15
         cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
             .8, (0, 255, 255), 2)

     # display the image to our screen
     cv2.imshow("Facial Recognition is Running", frame)
     key = cv2.waitKey(1) & 0xFF

     # quit when 'q' key is pressed
     if key == ord("q"):
         break

     # update the FPS counter
     fps.update()

 # stop the timer and display FPS information
 fps.stop()
 print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
 print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

 # do a bit of cleanup
 cv2.destroyAllWindows()
 vs.stop()
 ```


!!! warning "Attenzione!!!"
    Ha funzionato tutto SOLO installando tutto su Python 3.8 con uv...
