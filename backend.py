from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
import requests
import bs4

load_dotenv(find_dotenv())

response = requests.get("https://weblumo.com/lumo-ai/ia-atencion-al-cliente/")
soup = bs4.BeautifulSoup(response.text, 'html.parser')
doc = Document(
    page_content=soup.get_text()
)

# Inicializamos creador de vectores
embeddings = OpenAIEmbeddings()
# Cargamos los vectores
vector_store = FAISS.from_documents([doc], embeddings)
# Creamos un recuperador de vectores
retriever = vector_store.as_retriever()
# Creamos el prompt desde un template
template = """You are a chatbot having a conversation with a human.

    Given the following extracted parts of a website and a question, create a final answer.

    {context}

    Human: {question}
    Chatbot:"""
prompt = PromptTemplate(
    input_variables=["question", "context"], template=template
)

# Inicializamos el modelo de lenguaje
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# Creamos la cadena de chat
chain = {
    "context": retriever, 
    "question": RunnablePassthrough()
    } | prompt | llm | StrOutputParser()