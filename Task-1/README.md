# Task 1: Chat with your PDF (RAG Pipeline)

A Streamlit app that lets you upload a PDF and ask questions. Answers come
**strictly from the document**, with page numbers as sources.

**Live demo:** https://genai-task-cbbkdcswf62nfsqk2j8xbz.streamlit.app/

## Tech stack
Python, LangChain, ChromaDB, Streamlit, Groq (LLM), HuggingFace embeddings
(`all-MiniLM-L6-v2`), pypdf

## How it works
1. **Extract:** pypdf reads the PDF page by page.
2. **Chunk:** text is split by section heading, then by size if a section is long.
3. **Embed and store:** each chunk is embedded and saved in ChromaDB (cosine similarity).
4. **Retrieve:** the question is embedded and the 3 closest chunks are fetched.
5. **Answer:** the LLM answers using only those chunks, and the page numbers are shown.

## Edge cases handled
- **No relevant context:** if no chunk is close enough (distance > 0.8), the LLM is
  not called and the app replies "I couldn't find this in the document."
  The prompt also forbids outside knowledge, which prevents hallucinations.
- **API rate limits:** retry with exponential backoff (2s, 4s, 8s), then a friendly message.
- **Empty input:** empty questions are rejected.
- **Scanned or empty PDFs:** a clear error is shown instead of a crash.

## Run locally
```bash
git clone https://github.com/pragatiarya27/GenAI-Task
cd GenAI-Task/Task-1
pip install -r requirements.txt
```
Create a `.env` file (or set an environment variable):
```
GROQ_API_KEY=your_key_here
```
Then run:
```bash
streamlit run app.py
```
On Streamlit Community Cloud, add `GROQ_API_KEY` under **Settings → Secrets** instead.