# WiringPi

WiringPi (<http://wiringpi.com/>) è una libreria di interfacciamento con
il GPIO del Raspberry. Viene rilasciata tramite la licenza GNU LGPLv3 e
supporta un consistente numero di linguaggi di programmazione, in
maniera nativa (C, C++, RTB) o tramite wrappers (Python e altri...).
E' stata progettata in analogia al "Wiring System" di Arduino, da cui
il nome *WiringPi* per collegare il concetto di *Wiring System* con il
*Raspberry Pi*.

La libreria WiringPi include una utility a riga di comando denominata
**gpio** che può essere utilizzata per visualizzare e modificare le
impostazioni dei pin GPIO.

La libreria WiringPi e la sua estensione in Python saranno lo strumento
principale con cui interagiremo con il GPIO, da cui la necessità di
capire prima di cosa si tratta e installarla sul nostro Raspberry.

Per installarla basta eseguire il semplice comando:

``` bash
$ sudo apt install wiringpi
```

confermare con Yes tutte le richieste di APT e al termine testare la
buona riuscita dell'operazione.

!!! tip "Suggerimento"
    Prima di installare la libreria WiringPi con il comando precedente (e
    prima di installare qualunque pacchetto con APT), sarebbe saggio
    aggiornare il database dei pacchetti (update) e poi i pacchetti veri e
    propri del sistema (upgrade) in modo che tutto sia sincronizzato al
    meglio.

    I comandi da eseguire sono:

    ``` bash
    $ sudo apt update
    $ sudo apt upgrade
    ```


Per verificare che effettivamente sia andato tutto bene basta eseguire il comando

``` bash
$ gpio -v
```

che dovrebbe fornire una serie di informazioni su di esso.

Provare per credere!!!
