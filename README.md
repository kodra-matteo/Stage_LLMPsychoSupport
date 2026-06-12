Versione al 12/06
Database usato: ChromaDB, salva i criteri diagnostici in locale in modo che l'ai non debba andarseli a cercare, evitando cosi possibili allucinazioni dell'ai.
Cartella data: contiene file .txt contenenti i criteri diagnostici da dare all'ai, si possono aggiungere quanti .txt si vogliono tanto il programma andrà a prendere la cartella per intero e analizzerà 
                singolarmente i file (dividendo le nozioni in diversi file rendo il tutto piu chiaro invece che avere 1 solo file contente libri interi di criteri diagnostici)
database.py: file che permette di aggiornare le nozioni da dare all'ai, come citato prima possiamo aggiungere quanti .txt vogliamo nella cartella data, dopo aver fatto ciò dobbiamo runnare 
             questo file in modo che la memoria venga aggiornata
agents.py: corpo del programma, definisce gli agenti e costruisce il grafo. Definisce il modo in cui pensano, li fa lavorare separatamente e infine unisce i risultati secondo la regola di aggregazione selezionata
app.py: parte front-end del sistema, usa streamlit per gestire la parte grafica
