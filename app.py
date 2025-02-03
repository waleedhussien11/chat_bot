import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from pyngrok import ngrok  # Import ngrok

# Expose Ollama API using Ngrok
def start_ngrok():
    tunnel = ngrok.connect(11434)  # Expose port 11434
    return tunnel.public_url  # Get public URL

# Get the new public URL for Ollama
ollama_public_url = start_ngrok()

# Enhanced prompt to suppress reasoning
template = """
[SYSTEM]
You are an assistant. Do not show your reasoning process. Provide concise, well-organized answers in bullet points.

[CONTEXT]
{context}

[QUESTION]
{question}

[ANSWER]
""".strip()

# Initialize components with updated Ollama base URL
embeddings = OllamaEmbeddings(model="qwen2.5:1.5b")
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="qwen2.5:1.5b", base_url=ollama_public_url)  # Use Ngrok URL

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def split_text(documents):
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    ).split_documents(documents)

def index_docs(documents):
    vector_store.add_documents(documents)

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    context = "\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    raw_response = chain.invoke({"question": question, "context": context})
    
    # Clean the response to remove residual reasoning
    if "[ANSWER]" in raw_response:
        clean_response = raw_response.split("[ANSWER]")[-1].strip()
    else:
        clean_response = raw_response.strip()
    
    clean_response = clean_response.split("---")[0].strip()  # Remove any trailing artifacts
    return clean_response

# Streamlit UI
st.title("PDF Chat Assistant")

# Display the Ngrok-exposed Ollama API URL
st.info(f"Ollama API running at: {ollama_public_url}")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        docs = split_text(load_pdf(tmp.name))
    index_docs(docs)
    os.remove(tmp.name)

    if query := st.chat_input("Ask about the document"):
        with st.chat_message("user"):
            st.write(query)
        
        with st.spinner("Analyzing..."):
            relevant_docs = retrieve_docs(query)
            final_response = answer_question(query, relevant_docs)
        
        with st.chat_message("assistant"):
            st.markdown(f"**Final Response:**\n\n{final_response}", unsafe_allow_html=True)
