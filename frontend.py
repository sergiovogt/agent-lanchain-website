import streamlit as st
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from backend import chain

st.title("💬 Chatea con el contenido de una página web ")
st.subheader("🤖 Hazle preguntas al Blog de LUMO.AI")

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        response = chain.invoke(
            prompt
        )
        st.write(response)
