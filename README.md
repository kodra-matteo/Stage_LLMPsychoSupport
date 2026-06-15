Versione al 15/06
## 🗄️ Database e Memoria Locale
**Database utilizzato: ChromaDB**
Il sistema utilizza un database vettoriale per salvare i criteri diagnostici localmente. In questo modo, l'Intelligenza Artificiale consulta esclusivamente la letteratura medica fornita, evitando di "inventare" risposte (le cosiddette *allucinazioni dell'IA*) e garantendo la massima aderenza clinica.

---

## 📂 Struttura del Progetto

Di seguito la descrizione dei componenti principali del sistema:

* 📁 **`data/` (Cartella)**
  Contiene i file `.txt` con i criteri diagnostici da fornire all'IA. È possibile aggiungere quanti file si desidera: il programma leggerà automaticamente l'intera cartella e analizzerà i file singolarmente.
  > *Nota: Dividere le nozioni mediche in più file testuali (es. un file per il DSM-5, uno per l'ICD-11, ecc.) rende l'architettura molto più ordinata, scalabile e facile da gestire rispetto all'avere un singolo file gigante.*

* ⚙️ **`database.py`**
  È lo script necessario ad aggiornare le nozioni a disposizione dell'IA. Come citato sopra, ogni volta che vengono aggiunti o modificati dei file `.txt` all'interno della cartella `data`, è necessario eseguire (*runnare*) questo file per ricaricare e aggiornare la memoria del sistema.

* 🧠 **`agents.py`**
  Rappresenta il *corpo/motore* del programma. 
  Questo file definisce i vari agenti AI (i medici virtuali) e costruisce l'architettura a grafo (LangGraph). Stabilisce il loro modo di "pensare", li fa lavorare e ragionare in parallelo e, infine, raccoglie e unisce i loro pareri secondo la regola matematica di aggregazione selezionata.

* 💻 **`app.py`**
  Rappresenta il *front-end* del sistema. 
  Utilizza il framework **Streamlit** per gestire l'intera parte grafica, offrendo un'interfaccia web pulita, intuitiva e pensata appositamente per gli utenti finali (medici e psicologi).
