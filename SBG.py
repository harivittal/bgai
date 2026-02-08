import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
import dotenv 


# --- 1. CONFIGURATION & SECURITY ---
# This gets the exact folder where SBG.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(BASE_DIR, '.env')

print(f"--- Debugging ---")
print(f"Looking for .env at: {dotenv_path}")

if os.path.exists(dotenv_path):
    # Use the 'utf-8-sig' trick that worked in your test script
    with open(dotenv_path, "r", encoding="utf-8-sig") as f:
        load_dotenv(stream=f, override=True)
    print("✅ .env file opened and loaded.")
else:
    print(f"❌ ERROR: .env file NOT FOUND at {dotenv_path}")

# Retrieve variables from the environment
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Safety Check: Stop the server immediately if keys are missing
if not all([GEMINI_KEY, SUPABASE_URL, SUPABASE_KEY]):
    print("\n" + "!"*50)
    print("CRITICAL ERROR: Variables are empty in the environment!")
    print(f"DEBUG: GEMINI_KEY found? {bool(GEMINI_KEY)}")
    print(f"DEBUG: SUPABASE_URL found? {bool(SUPABASE_URL)}")
    print(f"DEBUG: SUPABASE_KEY found? {bool(SUPABASE_KEY)}")
    print("!"*50 + "\n")
    sys.exit(1)

# --- 2. INITIALIZATION ---
app = FastAPI()

# Initialize Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Embeddings (Used to convert questions into math vectors)
embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

# Initialize Gemini AI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    google_api_key=GEMINI_KEY
)

# --- 3. DATA MODELS ---
class QuestionRequest(BaseModel):
    question: str

# --- 4. API ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "Gita AI Backend is Online", "version": "1.0.0"}

@app.post("/ask")
async def ask_gita(request: QuestionRequest):
    try:
        # 1. Convert user question into an embedding
        query_embedding = embeddings_model.embed_query(request.question)

        # 2. Search Supabase for the most relevant Gita verses (using RPC)
        # Note: 'match_gita_contents' is the SQL function you ran in Supabase
        result = supabase.rpc(
            "match_gita_contents",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.3, # Adjust for strictness
                "match_count": 3        # Number of verses to retrieve
            }
        ).execute()

        if not result.data:
            return {"answer": "I couldn't find a specific verse related to that. Please try rephrasing."}

        # 3. Format the retrieved verses for the AI
        context = "\n\n".join([f"Verse: {item['content']}" for item in result.data])

        # 4. Ask Gemini to answer based ONLY on the 1972 Gita text
        prompt = (
            f"You are a spiritual assistant based on the 1972 Bhagavad Gita As It Is. "
            f"Using the following verses, answer the question accurately and compassionately.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {request.question}\n\n"
            f"Answer:"
        )
        
        response = llm.invoke(prompt)
        
        return {
            "answer": response.content,
            "verses": result.data  # Sending verses back so Flutter can show them
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. RUN SERVER (Local Testing) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)