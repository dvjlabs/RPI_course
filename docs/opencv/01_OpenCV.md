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

L'installazione della libreria su Raspbian Buster è notevolmente
semplificata dalla presenza di alcuni pacchetti precompilati nei
repository.

L'installazione descritta di seguito è relativo alla versione 3.4.x
nonostante sia già disponibile, anche su Raspbian, la versione 4.1.x.
Purtroppo nei nostri test di laboratorio abbiamo avuto qualche
problemino con essa...

Procedete come di seguito. Ricordate inoltre che qualsiasi operazione di
installazione del sistema inizia con l'aggiornamento dello stesso...

``` bash
$ sudo apt install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
$ sudo apt install libxvidcore-dev libx264-dev
$ sudo apt install libjpeg-dev libtiff5-dev libjasper-dev libpng-dev
$ sudo apt install libhdf5-dev libhdf5-serial-dev libhdf5-103
$ sudo apt install libatlas-base-dev gfortran
$ sudo apt install libfontconfig1-dev libcairo2-dev
$ sudo apt install libgdk-pixbuf2.0-dev libpango1.0-dev
$ sudo apt install libgtk2.0-dev libgtk-3-dev
$ sudo apt install libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
$ sudo apt install python3-dev
```

Poi, finalmente...

``` bash
$ sudo pip3 install opencv-contrib-python==3.4.3.18
```

## TEST

Al termine dell'installazione si può procedere con il primo test di
funzionamento, che verifica semplicemente se l'installazione è andata a
buon fine. Il test prova semplicemente a caricare il modulo `cv2`
all'interno dell'interprete `python3`.

``` bash
$ python3
Python 3.7.3 (default, Apr  3 2019, 05:39:12)
[GCC 8.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv2
>>>
```


## LAB 00 - Controlliamo un pò di versioni

Prima di procedere, assicuriamoci, ancora, di avere installato
correttamente le versioni di Opencv e di Python. Ancora , direte voi...
beh la prudenza non è mai troppa. Probabilmente troverete già installato
il programma Thonny sul vostro Raspberry. Thonny è un semplice ed
"abbastanza" completo IDE (Integrated Development Environment) per
sviluppare applicazioni in Python. Quindi, caricate su Thonny il testo
sotto indicato ed eseguitelo... Se non dovesse funzionare potete sempre
utilizzare la shell di comandi (terminale).

``` bash
import cv2 as cv
import sys
print("Versione Python")
print(sys.version)
print ("versione OpenCv")
print(cv.__version__)
```


## LAB 01 - Carichiamo un'immagine

Cominciamo a prendere un pò di confidenza con le immagini. Carichiamone
una e proviamo a comprendere il codice sotto riportato.

``` bash
import cv2
import numpy as np
from matplotlib import pyplot as plt
#legge un file in formato .jpg mostrato poi in due finestra
#una di questa mostra l'immagine in scala di grigi

img = cv2.imread('watch.jpg',cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('watch.jpg')
cv2.imshow('imageGray',img)
cv2.imshow('imageColor',img1)
#wait for a pressed key
cv2.waitKey(0)
cv2.destroyAllWindows()
```


## LAB 02 - Ancora sulle immagini

Ancora sulle immagini. Questa volta invece di usare una immagine
trasformata in scala di grigio, usiamo l'immagine a colori. Notate il
diverso tasto di controllo per uscire dal programma e la possibilità di
salvare l'immagine sotto un altro nome.

``` bash
import cv2 as cv
import sys
print("Versione Python")
print(sys.version)
print ("versione OpenCv")
print(cv.__version__)

#import numpy library to do some draws
import numpy as np
# Load an color image in grayscale
img = cv.imread('cam.jpg',cv.IMREAD_COLOR)
print ("immagine caricata")
cv.imshow('image', img)
#wait for a pressed key
k=cv.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv.imwrite('camsalvata.png',img)
    cv.destroyAllWindows()
```


## LAB 03 - Disegniamo qualcosa. Forme, parole, linee... liberate la vostra fantasia e create nuove immagini

``` bash
import numpy as np
import cv2

img = cv2.imread('watch.jpg',cv2.IMREAD_COLOR)
cv2.line(img,(0,0),(200,300),(255,255,255),5)
cv2.rectangle(img,(15,25),(200,250),(0,0,255),5)
cv2.circle(img,(100,63), 55, (0,255,0), -1)

pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
# OpenCV documentation had this code, which reshapes the array to a 1 x 2. I did not
# find this necessary, but you may:
#pts = pts.reshape((-1,1,2))
cv2.polylines(img, [pts], True, (0,255,255), 3)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'Ciao Mondo!',(0,50), font, 1, (200,255,155), 2, cv2.LINE_AA)
#now show the modified image
cv2.imshow('image',img)
#wait for a pressed key
cv2.waitKey(0)
cv2.destroyAllWindows()
```


## LAB 04 - Finalmente un pò di video

``` bash
import numpy as np
import cv2
#a little of videos
#reading video files
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read() #il modo più semplice per leggere un file video

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```


## LAB 05 - Apriamo qualche finestra

``` bash
import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #color = cv.cvtColor(frame, cv.COLOR_BRG)
    # Display the resulting frame
    cv.imshow('frame',gray)
    cv.imshow('frame1',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
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
