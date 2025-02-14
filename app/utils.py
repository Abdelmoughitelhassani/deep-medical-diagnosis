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
import langid
from langdetect import detect




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
                k=2,
                memory_key="chat_history",    # Key for storing conversation history
                return_messages=True          # Whether to return messages
            )



# Prompt pour vérifier si la question est pertinente
classification_prompt = """
You are an AI assistant that classifies user questions based on their relevance to medical imaging.
Your task is to determine if the question is related to chest X-ray, chest CT, brain MRI, or diseases related to these imaging techniques.

Question:
{question}

Instructions:
1. If the question is related to chest X-ray, chest CT, brain MRI, or diseases related to these imaging techniques, respond with: "VALID".
2. If the question is unrelated, respond with: "INVALID".

ONLY respond with "VALID" or "INVALID".
"""

# Création du prompt
classification_prompt_template = PromptTemplate(
    input_variables=["question"],
    template=classification_prompt,
)

# Chaîne de classification
classification_chain = classification_prompt_template | llm | StrOutputParser()


"""
Initialise la chaîne pour reformuler les questions contextuelles.
"""
# system_prompt = (
#     "Given a chat history and the latest user question, "
#     "which might reference context in the chat history, "
#     "reformulate the question to make it standalone, "
#     "so that it can be understood without the chat history. "
#     "If the question is already standalone and doesn't need reformulation, "
#     "just return the question as it is. "
#     "Do NOT provide any answer or explanation, just return the reformulated or original question."
# )

system_prompt = (
    "Given a chat history and the latest user question, "
    "which might reference context in the chat history, "
    "reformulate the question to make it standalone, "
    "so that it can be understood without the chat history. "
    "If the question is already standalone and doesn't need reformulation, "
    "just return the question as it is. "
    "If the user's question has no meaningful connection to the chat history, "
    "leave it exactly as it is without any modification. "
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







def translate_to_english(text):
    """Utilise le LLM pour traduire uniquement le texte en anglais."""
    translation_prompt = f"Translate the following text to English and return only the translated text without any extra explanation:\n\n{text}"
    response = llm.invoke(translation_prompt)  
    return response.content.strip()  # Extrait uniquement le texte traduit



def detect_language(text):
    """Détecte la langue avec langid."""
    lang = detect(text)  # Retourne ('fr', score)
    return lang

def detect_and_translate_input(user_input):
    """Détecte la langue et traduit en anglais si nécessaire."""
    detected_lang = detect_language(user_input)  # Utilise langid

    if detected_lang != "en":  # Traduire si ce n'est pas en anglais
        return translate_to_english(user_input), detected_lang

    return user_input, "en"  # Déjà en anglais, pas besoin de traduire






def translate_to_original_language(text, target_lang):
    """Traduit la réponse de l'anglais vers la langue de l'utilisateur."""
    translation_prompt = f"Translate the following text to {target_lang} and return only the translated text without any extra explanation::\n\n{text}"
    return llm.invoke(translation_prompt).content.strip()





def get_llm_response(user_input):
    """
    Génère une réponse du LLM pour une entrée utilisateur donnée,
    en s'assurant que le pipeline fonctionne en anglais.
    """
    # Détecter la langue et traduire en anglais
    translated_input, original_lang = detect_and_translate_input(user_input)

    print(translated_input, original_lang)

    # Reformuler la question (si nécessaire)
    chat_history = memory.load_memory_variables({}).get("chat_history", [])
    standalone_question = contextualize_chain.invoke({
        "input": translated_input,
        "chat_history": chat_history
    })

    print(standalone_question)


    # Vérifier si la question est pertinente
    classification_result = classification_chain.invoke({"question": standalone_question})

    if classification_result.strip() == "INVALID":
        return "I can only assist with questions related to healthcare, specifically chest X-ray, CT, brain MRI, and related diseases."
    
    # Obtenir la réponse du modèle
    response = rag_chain.invoke({
        "question": standalone_question,
        "chat_history": chat_history
    })

    response_text = response["answer"]

    print(response_text)

    # Si l'input n'était pas en anglais, traduire la réponse vers la langue d'origine
    if original_lang != "en":
        response_text = translate_to_original_language(response_text, original_lang)

    return response_text




# def get_llm_response(user_input):
#     """
#     Génère une réponse du LLM après avoir vérifié si la question est pertinente.
#     """
    

#     # Reformuler la question
#     chat_history = memory.load_memory_variables({}).get("chat_history", [])
#     standalone_question = contextualize_chain.invoke({
#         "input": user_input,
#         "chat_history": chat_history
#     })

#     print(standalone_question)
    
#     # Vérifier si la question est pertinente
#     classification_result = classification_chain.invoke({"question": standalone_question})

#     if classification_result.strip() == "INVALID":
#         return "I can only assist with questions related to healthcare, specifically chest X-ray, CT, brain MRI, and related diseases."
    
#     # Obtenir la réponse du modèle
#     response = rag_chain.invoke({
#         "question": standalone_question,
#         "chat_history": chat_history
#     })
    
#     return response["answer"]

    

# def get_llm_response(user_input):
#     """
#     Génère une réponse du LLM pour une entrée utilisateur donnée.
#     """
    
#     # Reformuler la question
#     chat_history = memory.load_memory_variables({}).get("chat_history", [])
#     standalone_question = contextualize_chain.invoke({
#         "input": user_input,
#         "chat_history": chat_history
#     })
    
    
#     response = rag_chain.invoke({
#         "question": standalone_question,
#         "chat_history": chat_history
#     })
#     return response["answer"]
