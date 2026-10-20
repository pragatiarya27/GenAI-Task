# Data Extraction Bot

An API that turns messy, unstructured text (resumes, invoices) into clean, validated JSON using an LLM.

## How it works

Messy text -> FastAPI endpoint -> LLM + Pydantic schema -> Validated JSON

1. A client sends text and a document type to `POST /extract`.
2. The Pydantic schema for that type is sent to the LLM with structured output (JSON schema).
3. The LLM replies in exactly that shape.
4. Pydantic validates the result again before returning it.

## Tech stack

- Python
- Pydantic (schemas and validation)
- Groq API with an OpenAI-compatible client (LLM with JSON-schema output)
- FastAPI and Uvicorn (API)

## Supported document types

- `resume`: name, email, phone, skills, years of experience
- `invoice`: invoice number, vendor, date, items, total amount, currency

## Run it locally

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"
uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs to test it in the browser.

## Example request

```json
{
  "text": "Rahul Mehta, rahul.m@example.com. Skills: Python, SQL. 3 years as a backend developer.",
  "doc_type": "resume"
}
```

## Example response

```json
{
  "name": "Rahul Mehta",
  "email": "rahul.m@example.com",
  "phone": null,
  "skills": ["Python", "SQL"],
  "years_of_experience": 3.0
}
```

## Design choices

- Missing fields come back as `null`, so the model never invents data.
- Unsupported document types return a clear `400` error before any LLM call is made.
- The API key is read from an environment variable and never stored in code.
- Adding a new document type needs only a new Pydantic schema and one line in `SCHEMAS`.

## Possible improvements

- PDF upload support
- Batch processing
- Logging to measure extraction accuracy
