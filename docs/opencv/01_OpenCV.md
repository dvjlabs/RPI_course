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
$ pip install opencv-contrib-python
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
    if cv2.waitKey(1) &amp; 0xFF == ord('q'):
        break

# release the camera from video capture
cap.release()
# De-allocate any associated memory usage
cv2.destroyAllWindows()
```

Se funziona tutto... benissimo!!! Puoi andare avanti Al prossimo capitolo!!! Se invece la videocamera non viene riconosciuta, ad esempio
la picamera sul Raspberry, occorre provare qualcosa di diverso!

Per il problema della picamera, riferirsi al software e agli esempi riportati qui:
<a href="https://www.raspberrypi.com/documentation/computers/camera_software.html" target="_blank">https://www.raspberrypi.com/documentation/computers/camera_software.html</a>



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

    if cv2.waitKey(1) & 0xff == ord('q'):
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

    if cv2.waitKey(1) & 0xFF == ord('q'):
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
    if cv2.waitKey(1) & 0xFF == ord('q'):
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

    if cv2.waitKey(1) & 0xFF == ord('q'):
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
$ pip install face-recognition
```
