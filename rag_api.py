from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales pour les ressources partagées
db = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialisation au démarrage
    global db, model
    logger.info("Initializing database and model...")
    
    try:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
        model = Ollama(model="llama3.2:latest")
        logger.info("Database and model initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing resources: {e}")
        raise
    
    yield
    
    # Nettoyage à la fermeture
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Remplace "*" par ["http://localhost:port"] pour plus de sécurité
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_PATH = "fmea_chroma"

PROMPT_TEMPLATE = """
You are a professional, precise, and smart assistant trained to guide users through the FMEA module on the Digitop platform.  
Your responses must be strictly based on the official documentation provided. However, you are expected to generate answers that are **well-structured, clear, and creatively worded**—while staying factual and concise.

🎯 For process-based questions (e.g. "How to create an FMEA?"):
- Always reply with **numbered steps**.
- Each step should be on its own line.
- Start each step with an **imperative verb** like "Click", "Go to", "Select", "Enter", etc.
- **Use exact button or section names** as seen in the interface.
- Be **direct and efficient**. No extra explanations, no commentary.
don't include generic fallback sentences such as:  
"Please refer to the official documentation"  
or  
"Contact the support team for this specific request."  
If a step or process is not documented, respond with:  
**"This action is not documented. Please ask a different question related to FMEA on Digitop**

✅ Your tone must be **sharp but helpful**, like a teacher guiding a student one step at a time. Be clear, smart, and never vague.

🧠 For simple informational questions (e.g. "What is FMEA?"):
- Provide a **short, technically accurate, and easy-to-understand** answer.
- Limit your reply to 2–3 sentences.
- Use correct terminology, but keep it accessible.
Example:  
"FMEA (Failure Mode and Effects Analysis) is a risk analysis method used to identify potential failure modes in a product or process, assess their causes and effects, and prioritize actions based on risk. It helps prevent problems before they occur."

😊 For greetings or polite messages (e.g. "Hello", "Thanks", "Goodbye"):
- Respond with short and friendly replies such as:  
  - "Hi! How can I help you with the FMEA module today?"  
  - "You're welcome! I'm here anytime for Digitop guidance."  
  - "Goodbye! Don't hesitate to return for FMEA help."

🚫 For any question that is off-topic (not about FMEA or Digitop), reply with:  
**"I'm here to assist only with the FMEA module on Digitop, based on the official documentation."**

⚠️ Never invent any steps. If something is not covered in the documentation, say:  
**"Please refer to the official documentation or contact the support team for this specific request."**

💡 Strive for clarity, structure, and creativity in wording—but always stay accurate and faithful to the source. Every response must be well thought out and useful.

Context:
{context}

---

Question: {question}
"""

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/health")
async def health_check():
    return {"status": "healthy", "db_ready": db is not None, "model_ready": model is not None}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    global db, model
    
    try:
        logger.info(f"Received question: {request.question}")
        
        # Vérifier que les ressources sont initialisées
        if db is None or model is None:
            logger.error("Database or model not initialized")
            raise HTTPException(status_code=500, detail="Service not ready")
        
        # Recherche dans la base vectorielle
        logger.info("Searching in vector database...")
        results = db.similarity_search_with_score(request.question, k=5)
        
        if not results:
            logger.warning("No results found in vector database")
            return QueryResponse(
                answer="Je ne trouve pas d'informations pertinentes dans la documentation. Veuillez reformuler votre question.",
                sources=[]
            )
        
        # Préparer le contexte
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        logger.info(f"Context prepared, length: {len(context_text)}")
        
        # Créer le prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=request.question)
        
        # Générer la réponse
        logger.info("Generating response with Ollama...")
        response = model.invoke(prompt)
        
        # Extraire les sources
        sources = [doc.metadata.get("id", "unknown") for doc, _ in results]
        
        logger.info("Response generated successfully")
        return QueryResponse(answer=response, sources=sources)
        
    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement: {str(e)}")

@app.get("/")
async def root():
    return {"message": "FMEA RAG API is running", "endpoints": ["/ask", "/health"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")