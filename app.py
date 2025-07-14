import streamlit as st
from query_data import query_rag

st.set_page_config(page_title="FMEA Assistant", layout="centered")

st.title("🤖 FMEA Assistant Chatbot")
st.markdown("Posez ici toutes vos questions sur la méthode **FMEA**. Le chatbot vous répondra uniquement à partir des documents indexés.")

# Champ de texte utilisateur
query = st.text_input("❓ Question :", placeholder="e.g. How to create an FMEA template?")

# Bouton de soumission
if st.button("🔍 Interroger la base FMEA") and query.strip() != "":
    with st.spinner("Recherche en cours..."):
        try:
            response = query_rag(query)
            st.success("✅ Réponse trouvée")
            st.markdown(f"**Réponse :**\n\n{response}")
        except Exception as e:
            st.error(f"Une erreur est survenue : {str(e)}")

st.markdown("---")
st.caption("© 2025 – FMEA Chatbot powered by LangChain & Ollama")
