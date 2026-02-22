from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from vectores import getVectores
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


llm = ChatOpenAI(model="gpt-5-nano")
prompt = ChatPromptTemplate.from_template("""
Tu es un assistant strictement basé sur les documents ci-dessous.
Ne réponds que si l'information est dans les documents.
Si la réponse n'est pas dans les documents, dis "Je ne sais pas".
<context>
{context}
</context>

Question: {input}
                                          
Réponse courte, précise et factuelle :
""")
vectores = getVectores()
retriever = vectores.as_retriever(search_type="mmr", search_kwargs={
                                  "k": 7, "fetch_k": 10, "lambda_mult": 0.5})
doc_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, doc_chain)
