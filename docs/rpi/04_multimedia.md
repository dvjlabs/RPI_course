# Multimedia

Quello che si vuole intendere in questa sezione come *Multimedia* è la
possibilità di utilizzare con il nostro Raspberry alcune periferiche che
poi torneranno utili nell\'utilizzo dell\'Intelligenza Artificiale.

Queste periferiche nello specifico sono:

-   Una Webcam USB (utile per il riconoscimento facciale)
-   Un microfono (per parlare, nella nostra dotazione è integrato nella
    webcam)
-   le casse audio (per ascoltare musica e\...la AI che parla!!!)

## Webcam

::: warning
::: title
Warning
:::

Per poter utilizzare la webcam (ovvero accedere al file della periferica
che la rappresenta) bisogna far parte del gruppo **video**. L\'utente
*pi* è automaticamente parte di questo gruppo. Se vuoi verificare
digita:

``` bash
$ groups
```

Verranno elencati i gruppi di cui l\'utente corrente fa parte. Se per
qualche motivo *video* non fosse fra questi, bisogna aggiungere il
proprio utente al gruppo con il comando:

``` bash
$ sudo usermod -a -G video NOMEUTENTE
```
:::

Per poter utilizzare la webcam, basta collegarla al Raspberry e
installare il software necessario al suo utilizzo:

``` bash
$ sudo apt install fswebcam
```

Questione di 1 minuto\...

Per fare una prima prova e farsi una foto con la webcam, basta digitare:

``` bash
$ fswebcam prova.jpg
```

Sorridete, poi aprite il file manager, andate a guardare la foto e
controllate il risultato :)

![Foto scattata con fswebcam](images/fswebcam_image.jpg)

Per modificare la risoluzione, si può agire con il parametro **-r**.
Basta non esagerare. Ad esempio:

``` bash
$ fswebcam -r 800x600 prova2.jpg
```

Se non vi piace la barra sotto (il *banner*), basta eliminarlo con
l\'opzione *\--no-banner*

``` bash
$ fswebcam -r 800x600 --no-banner prova3.jpg
```

Il risultato:

![Foto scattata con fswebcam](images/fswebcam_image_no_banner.jpg)

Più o meno tutto qui!

## Casse audio

Queste sono davvero semplici da provare! Collegate le casse al Raspberry
(jack e USB per l\'alimentazione) e provate a sentire un video di
Youtube! Se non funziona, forse il problema è che la periferica
collegata al jack non è stata riconosciuta. Per assicurarci della cosa,
proviamo con:

``` bash
$ sudo raspi-config

Advanced Options ---> Audio ---> Jack
```

E questo è tutto!

## Microfono integrato

Una volta positivamente testate le casse, si potrà provare anche con il
microfono: basterà registrare qualche secondo di conversazione e poi
provare a riascoltarla!

Per fare ciò, procediamo con le utility installate tramite pulseaudio.
Per registrare:

``` bash
$ arecord prova.wav
```

Dite una frase tipo \"che bello questo corso!!!\", aspettate un paio di
secondi e poi premete la combinazione di tasti **CTRL + C** per
interrompere la registrazione.

Per riascoltare le nostre soavi parole, digitate un bel:

``` bash
$ aplay prova.wav
```

E anche questa è fatta!!!
