import os
import shutil
from vectores import getVectores
from fastapi import FastAPI, UploadFile, File, HTTPException
from llm import chain
from fastapi import FastAPI
import uvicorn

app = FastAPI(
    title="RAGs API",
    version="1.0",
    description="API pour répondre aux questions sur les documents chargés."
)


@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API RAGs! Utilisez le endpoint /ask pour poser vos questions."}


@app.post("/ask")
def ask_question(question):
    reponse = chain.invoke({"input": question})
    return reponse["answer"]


@app.post("/upload_docs")
async def upload_file(file: UploadFile = File(..., description="Téléverser un document PDF ou TXT")):
    vectorstore = getVectores()

    DOCS_PATH = "./data/docs"

    os.makedirs(DOCS_PATH, exist_ok=True)
    try:
        # Sauvegarde sur disque
        file_path = os.path.join(DOCS_PATH, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Charger le nouveau document
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        if file.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)

        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(documents)

        # Ajouter au vectorstore et persister
        vectorstore.add_documents(docs)

        return {"status": "success", "file": file.filename, "documents_added": len(docs)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    config = uvicorn.Config("app:app", port=8000, log_level="info")
    server = uvicorn.Server(config=config)
    server.run()
