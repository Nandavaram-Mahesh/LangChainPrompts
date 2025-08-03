import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    
)

model = ChatHuggingFace(llm=llm)

chat_history = []


while True:
    
    user_input = input("You: ")
    chat_history.append(user_input)
    if  user_input=='exit':
        break
    
    response = model.invoke(chat_history)
    chat_history.append(response.content)
    
    print(f"Bot: {response.content}")

print(chat_history)


