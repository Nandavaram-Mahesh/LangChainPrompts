import os
import streamlit as st
# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# import template.json



load_dotenv()

# Get token from environment
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

if not hf_token:
    raise ValueError("Hugging Face token not found. Please set the HUGGINGFACEHUB_ACCESS_TOKEN environment variable.")


llm = HuggingFaceEndpoint(
    
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    
)

model = ChatHuggingFace(llm=llm)

st.header('Reasearch Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('template.json')



if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    st.write(result.content)