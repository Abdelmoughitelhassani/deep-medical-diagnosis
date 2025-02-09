import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import BaseMessage
from langchain.prompts import PromptTemplate

import logging
from dotenv import load_dotenv

# Charger le fichier .env
load_dotenv()

# Suppress logs from the specific module
logging.getLogger("langchain_community.vectorstores.pinecone").setLevel(logging.ERROR)


os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_ENVIRONMENT"] = os.getenv("PINECONE_ENVIRONMENT")




llm = ChatGroq(
        temperature=0,
        model="llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY") # Groq API Key
    )


model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Pinecone instance
retriever =  Pinecone.from_existing_index(
    index_name='test',
    embedding=model,
    text_key="content"
)


memory = ConversationBufferWindowMemory(
                k=5,
                memory_key="chat_history",    # Key for storing conversation history
                return_messages=True          # Whether to return messages
            )


"""
Initialise la chaîne pour reformuler les questions contextuelles.
"""
system_prompt = (
    "Given a chat history and the latest user question, "
    "which might reference context in the chat history, "
    "reformulate the question to make it standalone, "
    "so that it can be understood without the chat history. "
    "If the question is already standalone and doesn't need reformulation, "
    "just return the question as it is. "
    "Do NOT provide any answer or explanation, just return the reformulated or original question."
)
prompt_prep_input = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
contextualize_chain =  prompt_prep_input | llm | StrOutputParser()


# Chaîne RAG avec mémoire
#rag_chain = ConversationalRetrievalChain.from_llm(
#    llm=llm,
#    retriever=retriever.as_retriever(),
#    memory=memory
#)

# Définir un prompt pour structurer la réponse et gérer différents cas
prompt_template = """
You are a helpful assistant in the healthcare domain, specifically focused on the analysis of chest CT, X-ray, and brain MRI images.
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Instructions:
1. If the context provides sufficient information, answer the question concisely in 2-3 sentences.
2. If the context is insufficient but the question is related to the healthcare domain, especially chest CT, X-ray, or brain MRI, add general knowledge to provide a complete answer in 2-3 sentences.
3. If the question is unrelated to the healthcare domain (chest CT, X-ray, or brain MRI), respond ONLY with:
   "I can only assist with questions related to healthcare, specifically chest CT, X-ray, and brain MRI. Please provide a question within this scope."

DO NOT provide any additional explanation or context about unrelated topics.

Answer:
"""


# Créer un objet `PromptTemplate`
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Chaîne RAG avec mémoire et prompt
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt}  # Passer le prompt ici
)




    

def get_llm_response(user_input):
    """
    Génère une réponse du LLM pour une entrée utilisateur donnée.
    """
    
    # Reformuler la question
    chat_history = memory.load_memory_variables({}).get("chat_history", [])
    standalone_question = contextualize_chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    
    
    response = rag_chain.invoke({
        "question": standalone_question,
        "chat_history": chat_history
    })
    return response["answer"]
