import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader # <- Aggiunto DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# permette di aggiornare le nozioni da dare all'ai. Aggiungiamo nozioni nuove dentro la cartella data (sotto forma di file .txt)
# runnando database.py aggiorna la memoria della nostra ai che da ora contiene le nozioni aggiornate

# percorsi delle cartelle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")


def build_database():
    # elimina il vecchio database
    if os.path.exists(CHROMA_DIR):
        print("Trovato un database preesistente. Cancellazione in corso...")
        shutil.rmtree(CHROMA_DIR)
        print("Vecchio database cancellato.")

    print(f"Lettura di TUTTI i file .txt nella cartella {DATA_DIR}...")

    # cerca e legge tutti i file che finiscono con .txt dentro la cartella 'data'
    loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    docs = loader.load()

    # se non esistono file .txt resituisce errore
    if not docs:
        print("Errore: Nessun file .txt trovato nella cartella 'data'. Inserisci almeno un file.")
        return

    # Divide in testo in chunk da 500 caratteri l'uno cosi da non sovraccaricare il sistema
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    print(f"Trovati e fusi {len(docs)} documenti. Divisi in {len(splits)} frammenti.")
    print("Creazione dei vettori in corso...")

    # trasforma il testo in array di numeri cosi da renderlo leggibile per l'ai
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # salva nel database l'array di numeri appena creato
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print("Database globale creato con successo!")

if __name__ == "__main__":
    build_database()