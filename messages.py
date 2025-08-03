import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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

messages=[
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about LangChain')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)

# Output - messages
# [
    
# SystemMessage(content='You are a helpful assistant', additional_kwargs={}, response_metadata={}),

# HumanMessage(content='Tell me about LangChain', additional_kwargs={}, response_metadata={}), 

# AIMessage(content='<think>\n\nWe are going to explain what LangChain is, its purpose, key components, and use cases.\nLangChain is a framework designed to simplify the creation of applications using large language models (LLMs). \nIt provides tools and abstractions to manage interactions with language 
# models, chain together multiple components, and integrate with various data sources and tools.\n\nKey aspects of LangChain:\n\n1. **Components**: \n   - **Models**: 
# Wrappers for different LLMs (like OpenAI, Hugging Face, etc.) and embedding models.\n   - **Prompts**: Tools for prompt management, including prompt templates and output parsers.\n   - **Indexes**: Help in working with external data, including document loaders, text splitters, vector stores, and retrievers.\n   - **Memory**: For persisting state between calls (e.g., conversation history).\n   - **Chains**: Sequences of calls to components (models, APIs, etc.). Pre-built chains for common tasks.\n   - **Agents**: Use an LLM to decide a sequence of actions. They can use tools (like search, calculator, etc.) to gather information.\n\n2. **Use Cases**:\n  
#  - Building chatbots with memory and context.\n   - Question answering over documents.\n   - Summarization, translation, and other text generation tasks.\n   - Integration with external APIs and databases.\n\n3. **Benefits**:\n   - Modular and extensible.\n   - Simplifies complex workflows involving LLMs.\n   - Community support and active development.\n\nExample: A simple chain that uses a prompt template to generate a greeting.\n\nWe\'ll write a short example in Python to demonstrate.\n</think>\n\n\n**LangChain** is an open-source framework designed to simplify the development of applications powered by **large language models (LLMs)** like OpenAI\'s GPT, Anthropic\'s Claude, or Meta\'s Llama. It acts as a "glue" layer, enabling developers to integrate LLMs into workflows that involve data retrieval, reasoning, 
# and action-taking. Here’s a concise overview:\n\n---\n\n### **Key Features & Concepts**\n1. **Modular Building Blocks**  \n   LangChain provides reusable components 
# for:\n   - **Models**: Interfaces for LLMs (e.g., GPT-4) and embedding models.\n   - **Prompts**: Tools for dynamic prompt templating and optimization.\n   - **Chains**: Pre-built sequences of operations (e.g., "summarize → translate → output JSON").\n   - **Agents**: LLM-driven systems that use **tools** (APIs, databases, calculators) to perform tasks.\n   - **Memory**: Short/long-term state management (e.g., chat history).\n   - **Indexes**: Integrate external data via vector stores (e.g., ChromaDB, Pinecone) for retrieval-augmented generation (RAG).\n\n2. **Retrieval-Augmented Generation (RAG)**  \n   Enables LLMs to pull data from external sources 
# (e.g., documents, databases) for accurate, context-aware responses.  \n   *Example*: Answering questions about a PDF by first searching its content.\n\n3. **Agent-Based Workflows**  \n   LLMs act as "reasoning engines" that decide which tools to use dynamically.  \n   *Example*: An agent that checks the weather via an API → suggests clothing → books a ride.\n\n4. **Multi-Step Reasoning**  \n   Chains break complex tasks into smaller steps (e.g., "Analyze sentiment → extract keywords → generate report").\n\n---\n\n### **Use Cases**\n- **Chatbots** with contextual memory and document support.\n- **Document analysis**: Summarize, translate, or Q&A over PDFs/text.\n- **Structured data extraction**: Pull info from unstructured text into tables/JSON.\n- **Automated workflows**: Research agents, code generators, or content creators.\n- **Knowledge management**: Enterprise RAG systems for internal docs.\n\n---\n\n### **Popular Integrations**\n- **Vector Databases**: FAISS, Pinecone, 
# Chroma.\n- **Tools**: SerpAPI (search), Wikipedia, Python REPL, custom APIs.\n- **LLM Providers**: OpenAI, Anthropic, Hugging Face, Cohere.\n- **Frameworks**: Streamlit, Gradio, FastAPI for deployment.\n\n---\n\n### **Example Workflow**\n```python\nfrom langchain.chains import RetrievalQA\nfrom langchain.llms import OpenAI\n\n# 
# 1. Load documents → split → embed → store in vector DB.\n# 2. Create a retrieval chain:\nqa_chain = RetrievalQA.from_chain_type(\n    llm=OpenAI(),\n    retriever=vector_db.as_retriever()  # Searches docs for relevant snippets\n)\n# 3. Query:\nresponse = qa_chain.run("What\'s LangChain?")\n```\n\n---\n\n### **Why Use LangChain?**\n- **Saves time**: Abstracts LLM complexity; no need to reinvent RAG/agents.\n- **Scalability**: Handles state, tooling, and data integration.\n- **Community**: 70k+ GitHub stars, active contributors, and extensive docs.\n\n### **Challenges**\n- Steep learning curve for beginners.\n- Overhead for simple tasks (use LiteChain or direct API calls for trivial cases).\n\n---\n\n**Resources**:  \n- [GitHub](https://github.com/langchain-ai/langchain)  \n- [Documentation](https://python.langchain.com/)  \n- [Tutorials](https://www.langchain.com/langchain)  \n\nLangChain democratizes advanced LLM applications, making it essential for AI developers building contextual, data-aware systems. 🚀', additional_kwargs={}, response_metadata={})

# ]