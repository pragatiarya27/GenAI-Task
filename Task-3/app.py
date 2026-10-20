import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

# 1. Connect to Groq (key is read from an environment variable, never hardcoded)
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)
MODEL = "openai/gpt-oss-20b"

# 2. Schemas (the "contract" for the output)
class Resume(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    skills: List[str] = Field(default=[], description="List of technical skills")
    years_of_experience: Optional[float] = Field(default=None, description="Total years of work experience")

class Invoice(BaseModel):
    invoice_number: str = Field(description="Invoice number or ID")
    vendor_name: str = Field(description="Name of the seller or company")
    invoice_date: Optional[str] = Field(default=None, description="Invoice date in YYYY-MM-DD format")
    items: List[str] = Field(default=[], description="List of purchased item names")
    total_amount: float = Field(description="Final total amount as a number")
    currency: Optional[str] = Field(default=None, description="Currency code like INR or USD")

SCHEMAS = {"resume": Resume, "invoice": Invoice}

# 3. Reusable extraction function
def extract(text, schema_class):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Extract the details from the text. If a field is missing, use null. Do not guess."},
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_class.__name__.lower(),
                "schema": schema_class.model_json_schema(),
            },
        },
    )
    return schema_class.model_validate_json(response.choices[0].message.content)

# 4. FastAPI app
app = FastAPI(title="Data Extraction Bot")

class ExtractRequest(BaseModel):
    text: str
    doc_type: str  # "resume" or "invoice"

@app.get("/")
def home():
    return {"message": "Data Extraction Bot is running"}

@app.post("/extract")
def extract_endpoint(request: ExtractRequest):
    schema_class = SCHEMAS.get(request.doc_type.lower())
    if schema_class is None:
        raise HTTPException(status_code=400, detail="doc_type must be 'resume' or 'invoice'")
    return extract(request.text, schema_class)
