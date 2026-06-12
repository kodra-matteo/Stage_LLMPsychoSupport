from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import os


# stati dei vari layer
class ClinicalState(TypedDict):
    clinical_case: str
    structured_data: str
    decision_deductive: str
    decision_inductive: str
    decision_case_based: str
    final_output: str

# temperatura settata a 0.1 per dare meno spazio creativo possibile all'ai sulle decisioni
llm = ChatOllama(model="llama3", temperature=0.1)

# directory varie
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
# Chiediamo al DB di restituirci i 2 paragrafi più pertinenti in base ai sintomi
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# trasforma la cartella clinica data in input in qualcosa di schematico e riassuntivo
def layer1_structuring(state: ClinicalState):
    """Layer 1: Riformula l'input caotico in una lista clinica chiara"""
    print("-> Layer 1: Estrazione e strutturazione dei sintomi...")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Sei un assistente clinico. Estrai dal caso fornito: 1. Sintomi attivi, 2. Durata temporale, 3. Impatto sul funzionamento. Sii estremamente conciso e schematico."),
        ("human", "{case}")
    ])
    chain = prompt | llm
    res = chain.invoke({"case": state["clinical_case"]})
    return {"structured_data": res.content}

# definizione agenti rule-based
def agent_deductive(state: ClinicalState):
    """Layer 2 (A): Applica rigidamente i criteri Top-Down"""
    print("-> Layer 2: Agente Deduttivo in esecuzione...")
    context = "\n".join([doc.page_content for doc in retriever.invoke(state["structured_data"])])

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Sei uno psichiatra dal ragionamento DEDUTTIVO. Parti dalle linee guida diagnostiche e valuta se i sintomi soddisfano i criteri in modo rigido.\nUsa questo manuale:\n{context}\n\nRestituisci ESATTAMENTE e SOLO una lista numerata delle 3 diagnosi più probabili (es. 1. Nome, 2. Nome, 3. Nome)."),
        ("human", "Caso strutturato:\n{case}")
    ])
    chain = prompt | llm
    res = chain.invoke({"context": context, "case": state["structured_data"]})
    return {"decision_deductive": res.content}


def agent_inductive(state: ClinicalState):
    """Layer 2 (B): Cerca pattern temporali Bottom-Up"""
    print("-> Layer 2: Agente Induttivo in esecuzione...")
    context = "\n".join([doc.page_content for doc in retriever.invoke(state["structured_data"])])

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Sei uno psichiatra dal ragionamento INDUTTIVO. Parti dalla durata temporale e dall'evoluzione dei sintomi specifici per risalire alla patologia.\nUsa questo manuale:\n{context}\n\nRestituisci ESATTAMENTE e SOLO una lista numerata delle 3 diagnosi più probabili (es. 1. Nome, 2. Nome, 3. Nome)."),
        ("human", "Caso strutturato:\n{case}")
    ])
    chain = prompt | llm
    res = chain.invoke({"context": context, "case": state["structured_data"]})
    return {"decision_inductive": res.content}


def agent_case_based(state: ClinicalState):
    """Layer 2 (C): Valuta la somiglianza con il prototipo clinico"""
    print("-> Layer 2: Agente Case-Based in esecuzione...")
    context = "\n".join([doc.page_content for doc in retriever.invoke(state["structured_data"])])

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Sei uno psichiatra esperto che usa un approccio CASE-BASED. Valuta quanto il paziente assomiglia al quadro clinico prototipico classico, tralasciando i micro-dettagli tecnici.\nUsa questo manuale:\n{context}\n\nRestituisci ESATTAMENTE e SOLO una lista numerata delle 3 diagnosi più probabili (es. 1. Nome, 2. Nome, 3. Nome)."),
        ("human", "Caso strutturato:\n{case}")
    ])
    chain = prompt | llm
    res = chain.invoke({"context": context, "case": state["structured_data"]})
    return {"decision_case_based": res.content}

# layer che si occupa di unire i risultati degli agenti secondo la regola di aggregazione selezionata
def layer3_aggregation(state: ClinicalState):
    """Layer 3: Agente Giudice che applica le regole di aggregazione matematiche"""
    rule = state.get("selected_aggregation_rule", "majority")
    print(f"-> Layer 3: Esecuzione dell'aggregazione tramite protocollo: {rule.upper()}...")

    # Definiamo le regole matematiche che il Giudice dovrà applicare
    rule_instructions = {
        "majority": (
            "Majority/Plurality Voting: Guarda ESCLUSIVAMENTE la 1° scelta di ogni agente. "
            "La diagnosi che compare più volte al 1° posto vince. In caso di parità (3 diagnosi diverse al 1° posto), "
            "valuta le 2° scelte come spareggio. Spiega brevemente il conteggio."
        ),
        "borda": (
            "Rank Protocol (Borda Count): Assegna 3 punti alla 1° scelta, 2 punti alla 2° scelta, 1 punto alla 3° scelta di ogni agente. "
            "Somma i punti di ogni patologia tra tutti gli agenti. Quella con il punteggio totale più alto vince. "
            "Mostra esplicitamente i calcoli (es. Schizofrenia: 3 + 2 + 0 = 5 punti)."
        ),
        "disapproval": (
            "Disapproval Protocol: Cerca il consenso ed evita diagnosi controverse. "
            "Assegna +1 punto per ogni volta che una diagnosi compare in una classifica. "
            "TUTTAVIA, se una patologia viene menzionata da un agente ma NON compare PER NULLA nella Top 3 di un altro agente, "
            "riceve un 'Veto' (Disapprovazione = -2 punti). Vince la diagnosi con il punteggio finale più alto. Mostra i calcoli."
        )
    }

    instruction = rule_instructions.get(rule, rule_instructions["majority"])

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Sei l'Agente Giudice (Supervisore) di un comitato medico. Il tuo compito è unificare i pareri di 3 psichiatri applicando RIGIDAMENTE questa formula matematica:\n\n{instruction}\n\nAlla fine dei calcoli, scrivi chiaramente in grassetto 'DIAGNOSI FINALE AGGREGATA: [Nome Diagnosi]'."),
        ("human",
         "Ecco le 3 classifiche espresse dai medici:\n\nAgente Deduttivo:\n{deductive}\n\nAgente Induttivo:\n{inductive}\n\nAgente Case-Based:\n{case_based}")
    ])

    chain = prompt | llm
    res = chain.invoke({
        "instruction": instruction,
        "deductive": state["decision_deductive"],
        "inductive": state["decision_inductive"],
        "case_based": state["decision_case_based"]
    })

    return {"final_output": res.content}


# costruzione del grafo
graph_builder = StateGraph(ClinicalState)

# aggiungiamo i 5 Nodi creati sopra
graph_builder.add_node("layer1", layer1_structuring)
graph_builder.add_node("agent_deductive", agent_deductive)
graph_builder.add_node("agent_inductive", agent_inductive)
graph_builder.add_node("agent_case_based", agent_case_based)
graph_builder.add_node("layer3_aggregation", layer3_aggregation)

# definizione collegamenti (archi)
graph_builder.add_edge(START, "layer1")

# dal Layer 1 la palla passa a tutti e 3 gli agenti simultaneamente
graph_builder.add_edge("layer1", "agent_deductive")
graph_builder.add_edge("layer1", "agent_inductive")
graph_builder.add_edge("layer1", "agent_case_based")

# prima di avviare il layer 3 si aspetta l'esecuzione di tutti e 3 gli agenti
graph_builder.add_edge("agent_deductive", "layer3_aggregation")
graph_builder.add_edge("agent_inductive", "layer3_aggregation")
graph_builder.add_edge("agent_case_based", "layer3_aggregation")

graph_builder.add_edge("layer3_aggregation", END)

# compiliazione del multi-agente
app_graph = graph_builder.compile()

# test del sistema da terminale, serve per poter testare il sistema senza dover ogni volta caricare l'interfaccia grafica
''' 
if __name__ == "__main__":
    caso_test = "Paziente maschio, 24 anni. Da circa 5 settimane presenta deliri di persecuzione e allucinazioni uditive. Si è isolato in camera e ha smesso di andare al lavoro. Non ci sono precedenti di episodi depressivi o maniacali."

    # SELEZIONA LA REGOLA TRA: "majority", "borda", o "disapproval"
    regola_scelta = "borda"

    print(f"\n[START] Avvio Sistema Multi-Agente (Regola: {regola_scelta.upper()})...\n")

    final_state = app_graph.invoke({
        "clinical_case": caso_test,
        "selected_aggregation_rule": regola_scelta
    })

    print("\n" + "=" * 50)
    print(" DECISIONI DEI 3 GIUDICI (LAYER 2):")
    print("=" * 50)
    print(f" AGENTE DEDUTTIVO:\n{final_state['decision_deductive']}\n")
    print(f" AGENTE INDUTTIVO:\n{final_state['decision_inductive']}\n")
    print(f" AGENTE CASE-BASED:\n{final_state['decision_case_based']}\n")

    print("\n" + "=" * 50)
    print(" RISULTATO LAYER 3 (AGGREGAZIONE E CALCOLO):")
    print("=" * 50)
    print(final_state['final_output'])
'''
