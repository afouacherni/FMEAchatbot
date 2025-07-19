import os
import logging
import gc
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from contextlib import asynccontextmanager

# Import ta fonction d'embedding depuis ton fichier get_embedding_function.py
from get_embedding_function import get_embedding_function

# Désactiver télémétrie LangChain
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db = None
model = None
CHROMA_PATH = "fmea_chroma"

PROMPT_TEMPLATE = """
You are a professional assistant specialized in the FMEA module on Digitop.
Input Categories and Response Types
1. Greetings and Thanks
Input examples: "Hello", "Hi", "Good morning", "Thank you", "Thanks", "Goodbye"
Response format: Brief, friendly acknowledgment followed by offer to help
Example: "Hello! I'm here to help you with FMEA processes in Digitop. What would you like to know?"
2. Procedural Questions (Step-by-step guidance)
Input examples: "How to create FMEA template", "How to add failure modes", "Steps to complete risk assessment"
Response format:

Numbered step-by-step instructions
Use exact button names from documentation
Include imperative verbs
Be precise and actionable

Example format:

Click on "New Template" button
Enter template name in the "Template Name" field
Select project type from the dropdown menu
Click "Save" to create the template

3. Informational Questions (Simple definitions)
Input examples: "What is FMEA?", "What does RPN mean?", "Explain severity rating"
Response format:

Brief, direct definition based on RAG documents
No additional context unless specifically in the documentation

Semantic Understanding Rules

Recognize questions with similar meanings even if worded differently
Examples of equivalent questions:

"How to make new FMEA?" = "How to create FMEA template?"
"What's the process for adding risks?" = "How to add failure modes?"
"Steps to finish analysis" = "How to complete FMEA?"

Rules:
1. Answer ONLY using the information provided in the "Context" below. Do not rephrase or invent anything.
2. Provide the complete step-by-step answer as is, no truncation.
3. If the context does not contain the answer, reply exactly:
   "This action is not documented. Please ask a different question related to FMEA on Digitop."
4. Return the steps exactly as they appear in the context, without adding extra notes or comments.
5. Do not mention that the layout or wording may vary.
6.answer politely if user say thanks ,thank you or any other type of thanking

Context:
{context}

Question: {question}

Answer:
"""

# Models Pydantic
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

# Startup et lifespan FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, model
    logger.info("Initializing Chroma and Ollama model...")

    try:
        # Initialiser la base Chroma avec la fonction d'embedding
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

        # Debug: Vérifier si la base contient des documents
        num_docs = 0
        try:
            results = db.similarity_search_with_score("", k=1)
            num_docs = len(results)
        except Exception as e:
            logger.warning(f"Could not fetch documents from Chroma DB: {e}")

        logger.info(f"Chroma DB loaded with {num_docs} documents.")

        # Initialiser OllamaLLM sans paramètres interdits
        model = OllamaLLM(
            model="llama3.2:latest",
            num_ctx=1024,
            num_predict=256,
            temperature=0.3,
            keep_alive="5m"
        )

        logger.info("Resources loaded successfully.")
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise

    yield

    logger.info("Cleaning up resources...")
    cleanup_resources()

def cleanup_resources():
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("Memory cleaned.")
    except Exception as e:
        logger.warning(f"Cleanup error: {e}")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "db_ready": db is not None, "model_ready": model is not None}

@app.get("/db/status")
async def db_status():
    if db is None:
        return {"status": "Chroma DB not initialized"}
    try:
        results = db.similarity_search_with_score("", k=1)
        count = len(results)
        return {"status": "ok", "documents_found": count}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    global db, model
    try:
        if db is None or model is None:
            raise HTTPException(status_code=500, detail="Service not ready")

        logger.info(f"Received question: {request.question}")

        results = db.similarity_search_with_score(request.question, k=3)
        if not results:
            return QueryResponse(answer="No relevant info found in docs. Please ask something else.", sources=[])

        # Ne plus tronquer, prendre le contenu intégral
        context_text = "\n---\n".join([doc.page_content for doc, _ in results])

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=request.question)

        response_text = ""
        for chunk in model.stream(prompt):
            response_text += chunk

        sources = [doc.metadata.get("id", "unknown") for doc, _ in results]
        cleanup_resources()

        return QueryResponse(answer=response_text, sources=sources)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        cleanup_resources()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/ask/stream")
async def stream_response(request: QueryRequest):
    global db, model
    if db is None or model is None:
        raise HTTPException(status_code=500, detail="Service not ready")

    logger.info(f"Streaming answer for: {request.question}")

    results = db.similarity_search_with_score(request.question, k=3)
    if not results:
        return StreamingResponse(iter(["No relevant info found."]), media_type="text/plain")

    # Ne plus tronquer ici non plus
    context_text = "\n---\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=request.question)

    async def generate():
        try:
            for chunk in model.stream(prompt):
                yield chunk
        except Exception as e:
            yield f"\n[Error]: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/")
async def root():
    return {"message": "FMEA RAG API running", "endpoints": ["/ask", "/ask/stream", "/health", "/db/status"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
