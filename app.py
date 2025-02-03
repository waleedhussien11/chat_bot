import streamlit as st
import tempfile
import os
from pyngrok import ngrok
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Start ngrok
public_url = ngrok.connect(8501).public_url
st.sidebar.write(f"🔗 **Public Link:** [Click to Access]({public_url})")

# Enhanced prompt to suppress reasoning
template = """
[SYSTEM]
you are assistant for user so, dont make it look on your thinking and say figure out give your conclusion directly and show it on points and specificaly and provide your answer on points and organize way 

[CONTEXT]
{context}

[QUESTION]
{question}

[ANSWER]
""".strip()

# Initialize components
embeddings = OllamaEmbeddings(model="qwen2.5:1.5b")
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="qwen2.5:1.5b", base_url="http://192.168.1.100:11434")

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
    
    # Clean the response to remove any residual reasoning
    if "[ANSWER]" in raw_response:
        clean_response = raw_response.split("[ANSWER]")[-1].strip()
    else:
        clean_response = raw_response.strip()
    
    # Ensure the response is concise and does not contain reasoning
    clean_response = clean_response.split("---")[0].strip()  # Remove any trailing artifacts
    return clean_response

# Streamlit UI
st.title("PDF Chat Assistant")

# Add a toggle button to show/hide thinking
show_thinking = st.toggle("Show Thinking Process", value=False)

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

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
            if show_thinking:
                st.write("Thinking process is hidden by default. Enable 'Show Thinking Process' to view it.")
            else:
                st.markdown(f"**Final Response:**\n\n{final_response}", unsafe_allow_html=True)
