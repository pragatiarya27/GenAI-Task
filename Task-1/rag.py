import re
import time
import uuid

from pypdf import PdfReader
from groq import RateLimitError, APIConnectionError
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

MAX_DISTANCE = 0.8
SOURCE_MARGIN = 0.15
NOT_FOUND = "I couldn't find this in the document."
MODEL_NAME = "openai/gpt-oss-120b"

_embedder = None
_llm = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embedder


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=MODEL_NAME, temperature=0)   # reads GROQ_API_KEY from the environment
    return _llm


def extract_pages(pdf_file):
    """Read a PDF (file path or uploaded file) and return text per page."""
    reader = PdfReader(pdf_file)
    pages_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages_text.append({"page": i + 1, "text": text})
    if not pages_text:
        raise ValueError("No text found. This PDF may be a scanned image.")
    return pages_text


def build_docs(pages_text):
    """Split pages into chunks: by numbered heading first, then by size."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = []
    for p in pages_text:
        sections = re.split(r"\n(?=\d+\.\s)", p["text"])
        for section in sections:
            section = section.strip()
            if len(section) < 40:
                continue
            for chunk in splitter.split_text(section):
                docs.append(Document(page_content=chunk, metadata={"page": p["page"]}))
    if not docs:
        raise ValueError("The PDF has too little text to use.")
    return docs


def build_vectorstore(docs):
    """Embed the chunks and store them in a fresh in-memory ChromaDB."""
    return Chroma.from_documents(
        documents=docs,
        embedding=get_embedder(),
        collection_name=f"pdf_rag_{uuid.uuid4().hex[:8]}",
        collection_metadata={"hnsw:space": "cosine"},
    )


def call_llm_with_retry(prompt, max_retries=3):
    wait = 2
    for attempt in range(1, max_retries + 1):
        try:
            return get_llm().invoke(prompt).content
        except (RateLimitError, APIConnectionError):
            if attempt == max_retries:
                raise
            time.sleep(wait)
            wait *= 2


def answer_from_pdf(question, vectorstore):
    """Return (answer, pages). Answers ONLY from the PDF."""
    if not question or not question.strip():
        return "Please type a question.", []

    results = vectorstore.similarity_search_with_score(question, k=3)
    good = [(doc, score) for doc, score in results if score <= MAX_DISTANCE]
    if not good:
        return NOT_FOUND, []

    context = "\n\n".join(doc.page_content for doc, _ in good)
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: {NOT_FOUND}
Do not use any outside knowledge.

Context:
{context}

Question: {question}
"""
    try:
        answer = call_llm_with_retry(prompt)
    except Exception as e:
        return f"⚠️ The AI service is busy right now. Please try again in a minute. ({type(e).__name__})", []

    best = min(score for _, score in good)
    cited = [doc for doc, score in good if score <= best + SOURCE_MARGIN]
    pages = sorted({doc.metadata["page"] for doc in cited})
    return answer, pages
