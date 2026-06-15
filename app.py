import streamlit as st
from agents import app_graph

# 1. CONFIGURAZIONE DELLA PAGINA
st.set_page_config(
    page_title="Sistema di Supporto Diagnostico - Psicosi",
    page_icon="⚕️",
    layout="wide"
)

# 2. BARRA LATERALE (Spiegazioni per il Medico)
with st.sidebar:
    st.title("⚙️ Impostazioni del Consenso")
    st.markdown(
        "Seleziona il metodo matematico con cui il sistema valuterà i pareri indipendenti dei modelli per calcolare la diagnosi finale.")

    selected_rule = st.selectbox(
        "Metodo di Aggregazione:",
        options=["majority", "borda", "disapproval"],
        format_func=lambda x: {
            "majority": "Voto di Maggioranza",
            "borda": "Punteggio a Classifica (Borda)",
            "disapproval": "Ricerca del Consenso (Veto)"
        }[x]
    )

    # Spiegazioni mediche/semplificate delle regole
    if selected_rule == "majority":
        st.info(
            "**Come funziona:** Il sistema guarda solo la prima diagnosi proposta da ciascun modello. Vince la patologia che compare più volte al primo posto.")
    elif selected_rule == "borda":
        st.info(
            "**Come funziona:** Viene assegnato un punteggio pesato (3 punti alla prima scelta, 2 alla seconda, 1 alla terza). Vince la patologia con il punteggio totale più alto. Utile per trovare la diagnosi che mette maggiormente d'accordo tutti.")
    elif selected_rule == "disapproval":
        st.info(
            "**Come funziona:** Assegna punti alle patologie nominate, ma applica una grave penalità (Veto) se una diagnosi viene proposta da un modello ma è completamente scartata dagli altri. Evita le diagnosi controverse.")

    st.markdown("---")
    st.markdown("### 🧠 Modelli di Ragionamento Utilizzati")
    st.write(
        "Il sistema consulta tre intelligenze artificiali indipendenti, ciascuna istruita a usare un diverso approccio clinico:")
    st.caption("- **Deduttivo:** Applica rigidamente i criteri del DSM-5 (Top-Down).")
    st.caption("- **Induttivo:** Analizza la traiettoria e l'evoluzione temporale dei sintomi (Pattern Recognition).")
    st.caption(
        "- **Case-Based:** Cerca la somiglianza globale tra il paziente e il quadro clinico prototipico (Illness Script).")

# 3. CORPO PRINCIPALE DELL'APP
st.title("⚕️ Supporto Diagnostico: Spettro della Schizofrenia")
st.markdown("*Strumento di consultazione basato su DSM-5 e Intelligenza Artificiale Locale.*")

clinical_case_input = st.text_area(
    "📝 Inserisci i dati clinici e anamnestici del paziente:",
    value="Paziente maschio, 24 anni. Da circa 5 settimane presenta deliri di persecuzione e allucinazioni uditive. Si è isolato in camera e ha smesso di andare al lavoro. Non ci sono precedenti di episodi depressivi.",
    height=150
)

if st.button("Genera Ipotesi Diagnostica", type="primary"):

    if not clinical_case_input.strip():
        st.error("Per favore, inserisci un caso clinico.")
    else:
        # Spinner senza termini tecnici
        with st.status("Analisi clinica in corso...", expanded=True) as status:
            st.write("Esame della cartella clinica...")
            st.write("Consultazione indipendente dei tre modelli diagnostici...")

            inputs = {
                "clinical_case": clinical_case_input,
                "selected_aggregation_rule": selected_rule
            }

            final_state = app_graph.invoke(inputs)

            st.write(f"Calcolo della diagnosi finale tramite {selected_rule.upper()}...")
            status.update(label="Analisi Completata", state="complete", expanded=False)

        # 4. OUTPUT PER IL MEDICO

        # In primissimo piano: La Diagnosi Finale
        st.markdown("## 🏆 Diagnosi Finale Suggerita")
        st.success(final_state["final_output"])

        st.divider()

        # Le decisioni dei singoli in breve
        st.markdown("### ⚖️ Pareri dei Singoli Modelli")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Approccio Deduttivo")
            # Mostriamo solo la parte della CLASSIFICA estratta dal testo
            classifica_ded = final_state["decision_deductive"].split("CLASSIFICA:")[-1] if "CLASSIFICA:" in final_state[
                "decision_deductive"] else final_state["decision_deductive"]
            st.info(classifica_ded)

        with col2:
            st.subheader("Approccio Induttivo")
            classifica_ind = final_state["decision_inductive"].split("CLASSIFICA:")[-1] if "CLASSIFICA:" in final_state[
                "decision_inductive"] else final_state["decision_inductive"]
            st.info(classifica_ind)

        with col3:
            st.subheader("Approccio Case-Based")
            classifica_case = final_state["decision_case_based"].split("CLASSIFICA:")[-1] if "CLASSIFICA:" in \
                                                                                             final_state[
                                                                                                 "decision_case_based"] else \
            final_state["decision_case_based"]
            st.info(classifica_case)

        # Sezione XAI (Trasparenza) richiesta dal professore
        st.markdown("---")
        with st.expander("🔍 Esamina il Processo Decisionale e il Ragionamento (Trasparenza AI)"):
            st.markdown("#### 1. Estrazione dei Sintomi (Dati Strutturati)")
            st.write(final_state["structured_data"])

            st.markdown("#### 2. Ragionamento Clinico Esteso")
            st.markdown("**Modello Deduttivo:**")
            st.write(final_state["decision_deductive"])
            st.markdown("**Modello Induttivo:**")
            st.write(final_state["decision_inductive"])
            st.markdown("**Modello Case-Based:**")
            st.write(final_state["decision_case_based"])