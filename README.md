# MedioBanca
Script in python per analisi e visualizzazione dei dati contenuti nell'archivio storico di Mediobanca. 

Il contenuto della repository, che non contiene i file contenenti i dati di partenza nè le successive elaborazioni, può essere diviso in tre gruppi: 

1) MedioBanca_Cleaning_Preprocessing.py contiene le operazioni di pulizia e preprocessamento dei dati. In particolare ci si concentra sulla pulizia delle colonne testuali, sulla corretta formattazione
   delle colonne contenti date e numeri, sulla selezione delle colonne utili per le analisi correnti e sul tentativo di uniformare le diverse forme accreditate per una stessa azienda. La disambiguazione
   è stata tentata sia calcolando una matrice di distanza basata sulla distanza di Levenshtein, sia attraverso il calcolo della similarità del coseno. L'output dello script è un file csv che viene assunto
   come punto di partenza per le elaborazioni successive.

2) Geo.py contiene invece lo script per la generazione di una mappa HTML - map.html - su cui sono stati geolocalizzati i crediti concessi da Mediobanca e gli eventuali export che tali crediti hanno
   generato. La mappa è interattiva e ogni punto cliccabile apre un popup che consente l'esplorazione dei dati disponibili.

3) I restanti file, attualmente in fase di elaborazione, si concentrano sulla visualizzazione dei rapporti tra Mediobanca e le aziende clienti e tra queste ultime, come tali rapporti emergono dai dati di
   partenza. Gli output "8.png" in forma statica e "dynamic_graph.html" in forma interattiva, per quanto ancora abbozzati, mostrano la direzione verso cui tendere. In particolare attualmente si sta tentando
   l'addestramento di un modello NER per il riconoscimento automatico dei nomi di entità all'interno dei dati di partenza.
   
