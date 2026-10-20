# GenAI-Task


A collection of small, practical GenAI projects built with Python, LLMs, and modern tooling. Each task lives in its own folder with its own code and dependencies.

## Projects at a glance

| Task | Project | Objective | Key tech stack |
|------|---------|-----------|----------------|
| 1 | [Chat with your PDF (RAG)](./Task-1) | Upload a PDF and ask questions. Answers come strictly from the document, with page numbers as sources. | Python, LangChain, ChromaDB, Streamlit, Groq, HuggingFace embeddings |
| 3 | [Data Extraction Bot](./Task-3) | Turn messy text (resumes, invoices) into clean, validated JSON. | Python, Pydantic, FastAPI, Uvicorn, Groq |

**Live demo (Task 1):** https://genai-task-cbbkdcswf62nfsqk2j8xbz.streamlit.app/

## Repository structure

```
GenAI-Task/
├── Task-1/    # Chat with your PDF (RAG pipeline)
├── Task-3/    # Data Extraction Bot (FastAPI + Pydantic)
└── README.md  # You are here
```

## Getting started

```bash
git clone https://github.com/pragatiarya27/GenAI-Task
cd GenAI-Task
```

Both projects use the [Groq API](https://console.groq.com/). Get a key and set it as an environment variable:

```bash
export GROQ_API_KEY="your_key_here"
```

Then follow the setup for the task you want to run (below).

---

## Task 1: Chat with your PDF (RAG Pipeline)

A Streamlit app that lets you upload a PDF and ask questions. Answers come **strictly from the document**, with page numbers as sources.

### Tech stack
Python, LangChain, ChromaDB, Streamlit, Groq (LLM), HuggingFace embeddings (`all-MiniLM-L6-v2`), pypdf

### How it works
1. **Extract:** pypdf reads the PDF page by page.
2. **Chunk:** text is split by section heading, then by size if a section is long.
3. **Embed and store:** each chunk is embedded and saved in ChromaDB (cosine similarity).
4. **Retrieve:** the question is embedded and the 3 closest chunks are fetched.
5. **Answer:** the LLM answers using only those chunks, and the page numbers are shown.

### Edge cases handled
- **No relevant context:** if no chunk is close enough (distance > 0.8), the LLM is not called and the app replies "I couldn't find this in the document." The prompt also forbids outside knowledge, which prevents hallucinations.
- **API rate limits:** retry with exponential backoff (2s, 4s, 8s), then a friendly message.
- **Empty input:** empty questions are rejected.
- **Scanned or empty PDFs:** a clear error is shown instead of a crash.

### Run locally
```bash
cd Task-1
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

---

## Task 3: Data Extraction Bot

An API that turns messy, unstructured text (resumes, invoices) into clean, validated JSON using an LLM.

### Tech stack
Python, Pydantic (schemas and validation), Groq API with an OpenAI-compatible client (JSON-schema output), FastAPI and Uvicorn

### How it works

```
Messy text -> FastAPI endpoint -> LLM + Pydantic schema -> Validated JSON
```

1. A client sends text and a document type to `POST /extract`.
2. The Pydantic schema for that type is sent to the LLM with structured output (JSON schema).
3. The LLM replies in exactly that shape.
4. Pydantic validates the result again before returning it.

### Supported document types
- `resume`: name, email, phone, skills, years of experience
- `invoice`: invoice number, vendor, date, items, total amount, currency

### Run locally
```bash
cd Task-3
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"
uvicorn app:app --reload
```
Open http://127.0.0.1:8000/docs to test it in the browser.

### Example request
```json
{
  "text": "Rahul Mehta, rahul.m@example.com. Skills: Python, SQL. 3 years as a backend developer.",
  "doc_type": "resume"
}
```

### Example response
```json
{
  "name": "Rahul Mehta",
  "email": "rahul.m@example.com",
  "phone": null,
  "skills": ["Python", "SQL"],
  "years_of_experience": 3.0
}
```

### Design choices
- Missing fields come back as `null`, so the model never invents data.
- Unsupported document types return a clear `400` error before any LLM call is made.
- The API key is read from an environment variable and never stored in code.
- Adding a new document type needs only a new Pydantic schema and one line in `SCHEMAS`.

### Possible improvements
- PDF upload support
- Batch processing
- Logging to measure extraction accuracy

---

## Common design principles

- **Grounded outputs:** both projects avoid hallucination, by answering only from retrieved context (Task 1) or returning `null` for missing fields (Task 3).
- **Fail gracefully:** clear errors and friendly messages instead of crashes.
- **Secrets stay out of code:** API keys are always read from environment variables.
