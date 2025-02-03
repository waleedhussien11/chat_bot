import os
import tempfile
from server import Flask, request, jsonify
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

app = Flask(__name__)

# Initialize components
embeddings = OllamaEmbeddings(model="qwen2.5:1.5b")
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="qwen2.5:1.5b")

# Enhanced prompt template
template = """
[SYSTEM]
you are an assistant for the user, so don't make it look like you are thinking. Give conclusions directly and provide answers in an organized way.

[CONTEXT]
{context}

[QUESTION]
{question}

[ANSWER]
""".strip()

def load_pdf(file_path):
    """Load PDF content."""
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def split_text(documents):
    """Split documents into smaller chunks."""
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    ).split_documents(documents)

def index_docs(documents):
    """Index documents into the vector store."""
    vector_store.add_documents(documents)

def retrieve_docs(query):
    """Retrieve relevant documents based on the query."""
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    """Generate an answer based on retrieved documents."""
    context = "\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    raw_response = chain.invoke({"question": question, "context": context})
    
    # Clean response to remove unwanted reasoning
    clean_response = raw_response.strip()
    if "[ANSWER]" in clean_response:
        clean_response = clean_response.split("[ANSWER]")[-1].strip()
    
    return clean_response

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Handle PDF upload and indexing."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file.save(tmp.name)
            docs = split_text(load_pdf(tmp.name))
        
        # Index documents
        index_docs(docs)
        os.remove(tmp.name)

        return jsonify({"message": "PDF processed and indexed successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle user queries about the uploaded PDF."""
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        relevant_docs = retrieve_docs(question)
        final_response = answer_question(question, relevant_docs)
        
        return jsonify({"answer": final_response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
