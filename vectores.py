from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv(".env")

DATA_DIR = "./data/docs"
Persis_dir = "./data/chroma"


def getVectores():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    # Charger et découper les documents
    vectores = Chroma(collection_name="embeddings_collection",
                      embedding_function=embeddings, persist_directory=Persis_dir)
    if vectores._collection.count() == 0:
        print("Chargement des documents et création des vecteurs...")
        loader = PyPDFLoader(f"{DATA_DIR}/PhillippeGougler.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0)
        split_docs = text_splitter.split_documents(docs)
    # Créer les embeddings et la base de données vectorielle

        vectores.add_documents(split_docs)
    else:
        print("Vecteurs déjà présents, chargement depuis ChromaDB...")

    return vectores
