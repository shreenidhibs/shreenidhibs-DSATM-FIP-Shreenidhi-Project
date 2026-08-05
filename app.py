import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from agents.policy_agent import PolicyHandbookAgent
from tools.pdf_tool import UPLOAD_FOLDER, embed_pdf_document


app = FastAPI(
    title="Policy and Handbook Assistant",
    description="Institutional policy and handbook RAG assistant",
    version="1.0.0"
)


agent = PolicyHandbookAgent()


@app.get("/")
def home():
    return {
        "message": "Policy and Handbook Assistant is running",
        "documentation": "/docs"
    }


@app.post("/embed_pdf")
async def embed_pdf(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename was provided."
        )

    safe_filename = Path(file.filename).name

    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    destination = UPLOAD_FOLDER / safe_filename

    try:
        with destination.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file)

        result = embed_pdf_document(
            pdf_path=str(destination),
            filename=safe_filename
        )

        return {
            "message": "PDF uploaded and embedded successfully.",
            **result
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"PDF ingestion failed: {error}"
        ) from error

    finally:
        await file.close()


@app.get("/ask-policy/{query}")
def ask_policy(query: str):
    cleaned_query = query.strip()

    if not cleaned_query:
        raise HTTPException(
            status_code=400,
            detail="Please enter a valid policy question."
        )

    try:
        return agent.run(cleaned_query)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to answer the question: {error}"
        ) from error
