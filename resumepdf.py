
import os
import boto3
import json
from pypdf import PdfReader
from pydantic import BaseModel, ValidationError
from typing import List
from dotenv import load_dotenv
load_dotenv("myenv.env")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# =====================================================
# Create Bedrock Client
# =====================================================
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# =========================
# CONFIG
# =========================

MODEL_ID = "amazon.nova-micro-v1:0"

bedrock = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)


# =========================
# PYDANTIC MODELS
# =========================

class Experience(BaseModel):
    company: str
    years: float


class Resume(BaseModel):
    name: str
    email: str
    phone: str
    skills: List[str]
    experience: List[Experience]


# =========================
# PDF TEXT EXTRACTION
# =========================

def extract_text_from_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =========================
# RESUME PARSER
# =========================

def parse_resume(resume_text):

    prompt = f"""
You are a resume parser.

Extract information from the resume and return ONLY valid JSON.

Expected JSON format:

{{
    "name": "string",
    "email": "string",
    "phone": "string",
    "skills": ["skill1", "skill2"],
    "experience": [
        {{
            "company": "company_name",
            "years": 0
        }}
    ]
}}

Do not return explanations.
Do not return markdown.
Return JSON only.

Resume:
{resume_text}
"""

    try:

        response = bedrock.converse(
            modelId=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        output_text = response["output"]["message"]["content"][0]["text"]

        print("\nRaw Model Output:\n")
        print(output_text)

        # Convert model output to Python dict
        data = json.loads(output_text)

        # Validate against schema
        validated_resume = Resume.model_validate(data)

        return {
            "success": True,
            "data": validated_resume.model_dump()
        }

    except json.JSONDecodeError:

        return {
            "success": False,
            "error": "Invalid JSON returned by model"
        }

    except ValidationError as e:

        return {
            "success": False,
            "error": "Validation failed",
            "details": e.errors()
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    pdf_path = input("Enter resume PDF path: ").strip()

    try:

        resume_text = extract_text_from_pdf(pdf_path)

        result = parse_resume(resume_text)

        print("\nParsed Result:\n")
        print(json.dumps(result, indent=4))

        with open("parsed_resume.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

        print("\nOutput saved to parsed_resume.json")

    except FileNotFoundError:

        print("Error: PDF file not found.")

    except Exception as e:

        print(f"Unexpected error: {e}")