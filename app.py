import streamlit as st
from agents import app_graph

# configurazione pagina
st.set_page_config(
    page_title="Copilot Diagnostico - Spettro Schizofrenia",
    page_icon="🧠",
    layout="wide"
)

# BARRA LATERALE
with st.sidebar:
    st.title("⚙️ Protocolli di Coordinamento")

    st.markdown("### 🧮 Regola di Aggregazione (Layer 3)")
    # l'utente può scegliere la regola dal menu a tendina
    selected_rule = st.selectbox(
        "Seleziona come il Giudice valuterà i voti:",
        options=["majority", "borda", "disapproval"],
        format_func=lambda x: {
            "majority": "Majority / Plurality Voting",
            "borda": "Rank Protocol (Borda Count)",
            "disapproval": "Disapproval Protocol"
        }[x]
    )

    st.markdown("---")
    st.markdown("### 🤖 Ruoli degli Agenti (Layer 2)")
    st.info("**🕵️ Agente Deduttivo:**\nVerifica rigidamente i criteri diagnostici Top-Down dal manuale.")
    st.info("**📊 Agente Induttivo:**\nCerca pattern Bottom-Up basati su durata ed evoluzione dei sintomi.")
    st.info("**📖 Agente Case-Based:**\nValuta la somiglianza globale con il quadro prototipico del disturbo.")

# body della pagina
st.title("🧠 AI Copilot per la Diagnosi Differenziale")
st.markdown("*Supporto decisionale per disturbi dello spettro della schizofrenia e altri disturbi psicotici.*")
st.warning(
    "⚠️ **Disclaimer Medico:** Questo sistema genera ipotesi diagnostiche basate su LLM locali. La decisione finale spetta SEMPRE al clinico umano.")

# input del caso clinico, test case gia messo di default cosi da poter provare l'app senza riscrivere il caso
caso_default = "Paziente maschio, 24 anni. Da circa 5 settimane presenta deliri di persecuzione e allucinazioni uditive. Si è isolato in camera e ha smesso di andare al lavoro. Non ci sono precedenti di episodi depressivi."

clinical_case_input = st.text_area(
    "📝 Inserisci i dettagli del caso clinico:",
    value=caso_default,
    height=150
)

# Bottone di avvio
if st.button("🚀 Avvia Analisi Multi-Agente", type="primary"):

    if not clinical_case_input.strip():
        st.error("Per favore, inserisci un caso clinico.")
    else:
        # Usiamo st.status per creare un bel menu a tendina che mostra i caricamenti
        with st.status("🧠 Elaborazione del caso in corso...", expanded=True) as status:

            st.write("1️⃣ Layer 1: Strutturazione dei dati clinici...")
            # Prepariamo l'input per il nostro grafo
            inputs = {
                "clinical_case": clinical_case_input,
                "selected_aggregation_rule": selected_rule
            }

            # esecuzione grafo
            # Nota: poiché Llama 3 gira 5 volte, questo processo impiegherà 30-60 secondi a seconda del tuo PC
            final_state = app_graph.invoke(inputs)

            st.write("2️⃣ Layer 2: Consultazione parallela dei 3 Agenti Psichiatrici...")
            st.write(f"3️⃣ Layer 3: Aggregazione tramite regola '{selected_rule.upper()}'...")

            status.update(label="✅ Analisi Completata!", state="complete", expanded=False)

        # risultati a schermo

        # --- LAYER 1 ---
        with st.expander("📂 Dati Strutturati (Layer 1)", expanded=False):
            st.markdown(final_state["structured_data"])

        st.divider()

        # --- LAYER 2 (3 Colonne per i 3 Agenti) ---
        st.markdown("### ⚖️ Decisioni Parallele (Layer 2)")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🕵️ Deduttivo")
            st.info(final_state["decision_deductive"])

        with col2:
            st.subheader("📊 Induttivo")
            st.info(final_state["decision_inductive"])

        with col3:
            st.subheader("📖 Case-Based")
            st.info(final_state["decision_case_based"])

        st.divider()

        # --- LAYER 3 (Risultato Finale) ---
        st.markdown("### 🏆 Risultato Aggregazione (Layer 3)")
        st.success(final_state["final_output"])

        # (Opzionale) Spazio per l'Human-in-the-loop
        st.markdown("### 👨‍⚕️ Intervento Umano (Layer 4)")
        feedback = st.text_input("Commento o correzione del medico rispetto al risultato:")
        if feedback:
            st.write("*(In una versione futura, questo feedback aggiornerà i pesi degli agenti)*")