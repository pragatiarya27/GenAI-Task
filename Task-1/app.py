import os
import streamlit as st

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass   # running in Colab/locally: the key comes from the environment

import rag

st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF")

# Streamlit re-runs this file on every click, so we keep data in session_state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Sidebar: upload + process ----------
with st.sidebar:
    st.header("1. Upload your PDF")
    pdf = st.file_uploader("Choose a PDF file", type="pdf")

    if st.button("Process PDF"):
        if pdf is None:
            st.warning("Please choose a PDF first.")
        else:
            try:
                with st.spinner("Reading and indexing the PDF..."):
                    pages = rag.extract_pages(pdf)
                    docs = rag.build_docs(pages)
                    st.session_state.vectorstore = rag.build_vectorstore(docs)
                    st.session_state.messages = []
                st.success(f"Ready! {len(pages)} pages, {len(docs)} chunks.")
            except ValueError as e:        # scanned PDF / too little text
                st.error(str(e))

# ---------- Chat area ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("pages"):
            st.caption(f"📄 Source: page {', '.join(map(str, m['pages']))}")

question = st.chat_input("Ask a question about your PDF...")

if question:
    if st.session_state.vectorstore is None:
        st.warning("Please upload and process a PDF first (use the sidebar).")
    elif not question.strip():
        st.warning("Please type a question.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, pages = rag.answer_from_pdf(question, st.session_state.vectorstore)
            st.write(answer)
            if pages:
                st.caption(f"📄 Source: page {', '.join(map(str, pages))}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "pages": pages}
        )
