from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware # type: ignore
import os

from dotenv import load_dotenv
from rag_engine.pinecone_db import init_pinecone, search_similar
from rag_engine.embedder import get_embedding_model
from rag_engine.generator import generate_response as generate_answer

load_dotenv()
init_pinecone()
embedder = get_embedding_model()

app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask_question(payload: Query):
    query = payload.query
    context = "\n\n".join(search_similar(
        os.getenv("PINECONE_INDEX_NAME"), query, embedder))
    answer = generate_answer(query, context)
    return {"query": query, "answer": answer}
