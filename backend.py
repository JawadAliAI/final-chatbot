import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# import pinecone
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Click Media Lab Chatbot")

# Load environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")


# Request body schema
class QueryRequest(BaseModel):
    question: str

# # Initialize Pinecone
# pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
# index = pinecone.Index(pinecone_index_name)


from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text])[0].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [embedding.tolist() for embedding in self.model.encode(texts)]

embedding_model = SentenceTransformerEmbeddings("BAAI/bge-small-en-v1.5")


# load the existin gindex
from langchain_pinecone import PineconeVectorStore

docseach = PineconeVectorStore.from_existing_index(
    index_name=pinecone_index_name,
    embedding=embedding_model
)

retriever = docseach.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Prompt template
prompt = PromptTemplate.from_template("""
You are a helpful assistant for a digital marketing company.
Try to answer the user's question based on the provided context from the company document.
If the answer is not found in the context, provide a helpful and accurate answer from your own knowledge, focusing on digital marketing topics.

Context:
{context}

Question:
{question}
""")

# Set up OpenAI LLM and QA chain
llm = ChatOpenAI()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
)



from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or better: ["http://localhost:3000"] or your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
async def read_root():
    return {"message": "Welcome to the Click Media Lab Chatbot API!"}

@app.post("/")
async def root():
    return {"message": "Welcome to the Click Media Lab Chatbot API!"}

# Chatbot route
@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        print("Received question:", request.question)
        result = qa_chain.invoke(request.question)
        return {"answer": result["result"]}
    except Exception as e:
        print("Error occurred:", str(e))  # This will show in your terminal
        return {"error": "Internal Server Error", "details": str(e)}
