# Collegamento da remoto

Una volta che hai sistemato il raspberry e hai effettuato
l'installazione del sistema operativo e dato un'occhiata al tutto,
potrebbe tornarti utile **non** scollegare ogni volta mouse, tastiera,
monitor e tutto il laboratorio, ma semplicemente collegare il raspberry
alla rete e all'alimentazione e collegartici **da remoto!**

I metodi più utilizzati per la connessione remota al raspberry sono
sostanzialmente 3: elencherò il nome comune della tecnologia, il
software (server) che va installato su raspberry per renderlo
disponibile e il client da utilizzare su Windows 10 per la connessione
remota, visto che a scuola abbiamo PC con Windows 10. Se avete
dispositivi con sistemi operativi Mac o Linux, documentatevi su Internet
sui client per la corrispondente tecnologia.


Protocollo  | Tipologia  | Server (su RPI)  | Client (su Win10)
------------|------------|------------------|-------------------
RDP         | Grafica    | Xrdp             | Remote Desktop
VNC         | Grafica    | vnc              | VNC Viewer
SSH         | Testuale   | sshd             | Putty

!!! note "Nota bene"
    Qualsiasi metodo sceglierai, ricordati che avrai bisogno di conoscere il
    **nome** e/o l'**indirizzo IP** del tuo raspberry!

    Cerca di capire **prima** come sia possibile ottenere (e magari
    modificare) queste informazioni!



## RDP

**Remote Desktop Protocol** è un protocollo di rete proprietario
sviluppato da Microsoft, che permette la connessione remota da un
computer a un altro in maniera grafica. Il protocollo di default
utilizza la porta TCP e UDP 3389.

I client RDP esistono per la maggior parte delle versioni di Microsoft
Windows, Linux, Unix, macOS, Android, iOS. I server RDP ufficiali
esistono per i sistemi operativi Windows nonostante ne esistano anche
per i sistemi Unix-Like.

!!! tip "Su RPi"
    Installa il servizio xrdp:

    ``` bash
    $ sudo apt install xrdp
    ```

    Fatto questo, riavvia.



!!! tip "Su Windows"
    Non devi fare nulla! Ti basta cercare il software
    *Connessione a Desktop Remoto*


## VNC

**Virtual Network Computing** è un protocollo per applicazioni software
di controllo remoto, utilizzato per amministrare il proprio computer a
distanza. Può essere utilizzato anche per controllare in remoto server
che non posseggono né monitor né tastiera.

Il protocollo di comunicazione usato a livello di trasporto è il TCP
sulla porta di default 5900, oppure tramite interfaccia HTTP sulla porta
5800/tcp.

!!! tip "Su RPI"

    Il server VNC è disponibile di default su Raspbian, ma va abilitato
    tramite **raspi-config**: Interfacing Options --> VNC --> Enable

    Fatto questo, riavvia.


!!! tip "Su Windows"
    Un client VNC gratuito è il VNC Viewer di RealVNC:
    <https://www.realvnc.com/en/connect/download/viewer/windows/>

    Scaricalo, installalo su Windows e provalo.



## SSH

**Secure Shell** è un protocollo che permette di stabilire una sessione
remota cifrata tramite interfaccia a riga di comando con un altro host
di una rete informatica. È il protocollo che ha sostituito l'analogo,
ma insicuro, Telnet, perché basato su una comunicazione **non** cifrata.

A livello server utilizza la porta 22, sia tramite TCP che UDP.

!!! tip "Su RPI"
    Il server SSH è disponibile di default su Raspbian, ma va abilitato
    tramite **raspi-config**: Interfacing Options --> SSH --> Enable

    Fatto questo, riavvia.


!!! tip "Su Windows"
    Ti basta scaricare Putty e usarlo senza neanche installarlo!

    Il sito ufficiale è: <https://www.putty.org/>
