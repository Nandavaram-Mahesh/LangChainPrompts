import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    
)

model = ChatHuggingFace(llm=llm)

chat_history = [SystemMessage(content='You are a helpful AI assistant')]


while True:
    
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if  user_input=='exit':
        break
    
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    
    print(f"Bot: {response.content}")

print(chat_history)

# Output 

# [
# SystemMessage(content='You are a helpful AI assistant', additional_kwargs={}, response_metadata={}),
   
# HumanMessage(content='which is bigger 2 or 0', additional_kwargs={}, response_metadata={}), 

# AIMessage(content='<think>\nFirst, the user asked: "which is bigger 2 or 0". This is a very basic math question. I need to compare the numbers 2 and 0.\n\nRecall: in the number line, numbers increase as you move to the right. So, negative numbers are smaller, zero is in the middle, and positive numbers are larger. Specifically, 0 is the origin, and 2 is a positive number to the right of 0.\n\nTherefore, 2 is greater than 0.\n\nThe question is "which is bigger", 
# meaning I should state which one is larger.\n\nPossible answers:\n\n- 2 is bigger than 0.\n\n- 2 is greater than 0.\n\nSince it\'s a direct comparison, I can simply 
# say "2 is bigger."\n\nBut to be complete, I should phrase it as a sentence.\n\nThis is likely a very straightforward query, perhaps from someone learning numbers or 
# testing the AI. I should answer clearly and without confusion.\n\nAdditional context: In mathematics, bigger often means greater in magnitude, but for real numbers, 
# it usually refers to value. Here, both are positive, but 0 has no magnitude? No, magnitude is absolute value, but "bigger" in common usage for numbers means larger numerical value.\n\nFor example:\n\n- Is 5 bigger than 3? Yes.\n\n- Is -2 bigger than -5? No, because -2 is to the right, so it\'s larger in value.\n\n- But magnitude: | -2 | = 2 and | -5 | = 5, so 5 has a larger magnitude, but -5 is smaller in value.\n\nIn this case, since 2 and 0 are both non-negative, and 2 > 0, there\'s no issue.\n\nConfirming: 2 > 0, so 2 is bigger.\n\nAnswer structure: Be direct, polite, and educational.\n\nFinal response: "2 is bigger than 0."\n\nTo make it engaging: 
# "The number 2 is bigger than 0."\n\nBut since the user said "which is bigger", I can say "2 is bigger."\n\nI think that\'s fine.\n</think>\n2 is bigger than 0. \n\nIn numerical terms:\n- 2 is a positive number.\n- 0 is zero, which is neither positive nor negative.\n- On a number line, 2 is to the right of 0, meaning it is larger.\n\nSo, the bigger number is **2**.', additional_kwargs={}, response_metadata={}), HumanMessage(content='now mulitpy bigger by 10', additional_kwargs={}, response_metadata={}), AIMessage(content='<think>\nWe determined that 2 is bigger than 0.\n Now, we are to multiply the bigger number (which is 2) by 10.\n So, 2 multiplied by 10 is 20.\n</think>\nBased on our previous comparison:  \n- **2 is bigger than 0**.  \n\n**Multiplication step**:  \nMultiply the bigger number (2) by 10:  \n**2 × 
# 10 = 20**.  \n\n**Final result**:  \nThe answer is **20**.', additional_kwargs={}, response_metadata={}),

# HumanMessage(content='what will be the remainder if we divide the result by 2', additional_kwargs={}, response_metadata={}), 

# AIMessage(content='<think>\n\nWe are given the result from the previous step, which is 20.\n The task is to find the remainder when 20 is divided by 2.\n\n We know that 20 divided by 2 is 10, which is an integer. \n Since 20 is exactly divisible by 2, the remainder is 0.\n\n Alternatively, we can use the modulus operation:\n 20 % 2 = 0\n\n Therefore, the remainder is 0.\n</think>\n\n\nThe result from the previous step is **20**.  \n\n**Division step**:  \nDivide 20 by 2:  \n**20 ÷ 2 = 10** (which is exact, with no remainder).  \n\n**Remainder calculation**:  \nSince 20 is **divisible by 
# 2** (i.e., \\( 20 = 2 \\times 10 + 0 \\)), the remainder is **0**.  \n\n**Final answer**:  \nThe remainder is **0**.', additional_kwargs={}, response_metadata={}), 

# HumanMessage(content='exit', additional_kwargs={}, response_metadata={})]