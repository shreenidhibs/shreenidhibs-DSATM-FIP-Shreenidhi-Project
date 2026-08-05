from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent.parent

UPLOAD_FOLDER = PROJECT_ROOT / "uploaded_pdfs"
DATABASE_FOLDER = PROJECT_ROOT / "chroma_policy_db"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
DATABASE_FOLDER.mkdir(parents=True, exist_ok=True)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_store = Chroma(
    collection_name="institution_policy_documents",
    embedding_function=embeddings,
    persist_directory=str(DATABASE_FOLDER)
)


def embed_pdf_document(
    pdf_path: str,
    filename: str
) -> dict[str, Any]:
    """
    Load an institutional policy PDF, divide it into chunks,
    generate embeddings and store them in ChromaDB.
    """

    file_path = Path(pdf_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    loader = PyPDFLoader(str(file_path))
    pages = loader.load()

    if not pages:
        raise ValueError(
            "No readable text was found in the uploaded PDF."
        )

    for page in pages:
        page.metadata["document_name"] = filename

        page_number = page.metadata.get("page")

        if isinstance(page_number, int):
            page.metadata["page"] = page_number + 1

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError(
            "The PDF could not be divided into readable chunks."
        )

    document_ids = []

    for index, chunk in enumerate(chunks):
        page_number = chunk.metadata.get("page", 0)

        document_ids.append(
            f"{filename}-page-{page_number}-chunk-{index}"
        )

    vector_store.add_documents(
        documents=chunks,
        ids=document_ids
    )

    return {
        "filename": filename,
        "pages_processed": len(pages),
        "chunks_created": len(chunks)
    }


def retrieve_data_from_pdf(
    query: str,
    number_of_results: int = 5
):
    """
    Search the uploaded institutional policy documents before using
    any external information source.

    This tool searches documents such as:

    - Student handbook
    - Attendance policy
    - Sick-leave and medical-leave policy
    - Examination regulations
    - Hall-ticket eligibility rules
    - Hostel regulations
    - Student disciplinary rules

    Use this tool first for questions about attendance eligibility,
    attendance percentage, sick leave, medical leave, condonation,
    examinations, hall tickets, malpractice, hostel curfew, fines,
    restricted items and other institutional regulations.

    It returns the most relevant institutional document chunks together
    with relevance scores. The agent should answer only from these
    retrieved chunks when the requested information is clearly present.
    """

    cleaned_query = query.strip()

    if not cleaned_query:
        return []

    number_of_results = max(
        1,
        min(number_of_results, 10)
    )

    return vector_store.similarity_search_with_relevance_scores(
        query=cleaned_query,
        k=number_of_results
    )
