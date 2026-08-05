Policy & Handbook Assistant - Complete Setup Guide
Objective:
Build a single-agent RAG application that answers institutional policy questions from uploaded PDFs (student handbook, leave policy, exam rules, hostel rules). It searches the uploaded documents first and only falls back to web search if required.

1. Project Structure
Newproject/
├── agents/
│   ├── __init__.py
│   └── policy_agent.py
├── tools/
│   ├── __init__.py
│   └── pdf_tool.py
├── uploaded_pdfs/
├── chroma_policy_db/
├── app.py
├── requirements.txt
└── .env
2. Create Virtual Environment
python3 -m venv .venv
source .venv/bin/activate
3. Install Packages
pip install fastapi uvicorn python-dotenv python-multipart
pip install langchain langchain-community langchain-text-splitters
pip install langchain-groq langchain-chroma langchain-huggingface
pip install chromadb sentence-transformers pypdf
pip install torch==2.2.2 transformers==4.41.2 sentence-transformers==3.0.1
pip install numpy==1.26.4 scipy==1.12.0
4. Create .env
GROQ_API_KEY=<your_groq_api_key>
5. Implement Files
1. tools/pdf_tool.py
   - Create ChromaDB
   - Embed uploaded PDF
   - Implement retrieve_data_from_pdf()

2. agents/policy_agent.py
   - Create PolicyHandbookAgent
   - Retrieve chunks
   - Ask Groq using only retrieved context
   - Return NOT_FOUND if answer absent

3. app.py
   - POST /embed_pdf
   - GET /ask-policy/{query}
6. Start Server
python -m uvicorn app:app --reload --port 8001
7. Open Swagger
http://127.0.0.1:8001/docs
8. Upload Handbook
POST /embed_pdf
Try it Out → Choose handbook PDF → Execute
9. Ask Questions
GET /ask-policy/{query}

Examples:
How many sick leaves can a student take?
What is the hostel curfew?
What is the minimum attendance requirement?
10. Common Errors & Fixes
• Attribute 'app' not found → app.py must contain app = FastAPI()
• Cannot import PolicyHandbookAgent → class missing in policy_agent.py
• Cannot import retrieve_data_from_pdf → function missing in pdf_tool.py
• Address already in use → kill process or use another port
• Torch/Transformers error → install compatible versions listed above
• ModuleNotFoundError → install missing package inside active virtual environment
11. Deliverables
✓ PDF uploaded through /embed_pdf
✓ retrieve_data_from_pdf updated
✓ Existing RAG loop reused
✓ /ask-policy/{query} endpoint
✓ Chroma vector database
✓ Policy answers generated from uploaded handbook
