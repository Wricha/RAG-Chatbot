from langchain_groq import ChatGroq
import os
import requests
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

memory = ConversationBufferMemory(return_messages=True)
def generate_response(query, context):

    history = memory.load_memory_variables({})["history"]
    messages = [
    {
        "role": "system",
        "content": (
            "You are CareerBuddy, a smart and helpful career guidance assistant. "
            "Use ONLY the context provided to answer the user's question. "
            "Be concise and relevant."
        )
    }
]

    for message in history:
        if isinstance(message, HumanMessage):
            messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            messages.append({"role": "assistant", "content": message.content})

    messages.append({
    "role": "user",
    "content": f"Context: {context}\n\nQuestion: {query}"
})

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization":f"Bearer {GROQ_API_KEY}",
            "Content-Type":"application/json"
        },
        json={
            "model":"gemma2-9b-it",
            "messages":messages,
            "max_tokens": 1000,
            "temperature": 0.74
        }
    )
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"]
    answer = answer.replace("*", "").replace("**", "")

    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(answer)

    return answer