import os
import tempfile
from flask import Flask, request, jsonify
import chromadb
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

app = Flask(__name__)

# Initialize ChromaDB and LangChain components
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

embeddings = OllamaEmbeddings(model="qwen2.5:1.5b")
model = OllamaLLM(model="qwen2.5:1.5b")

# Enhanced prompt
template = """
[SYSTEM]
You are an assistant for the user. Provide answers in a direct, structured, and organized way.

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

def index_docs(documents, category):
    """Store documents in ChromaDB with category metadata."""
    for doc in documents:
        text = doc.page_content
        embedding = embeddings.embed_query(text)  # Convert text into embedding
        doc_id = str(hash(text))  # Generate a unique ID for each document
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[{"category": category, "text": text}]
        )

def retrieve_docs(query, category):
    """Retrieve relevant documents from ChromaDB filtered by category."""
    query_embedding = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5  # Retrieve top 5 relevant documents
    )

    # Filter by category
    relevant_docs = []
    for i, metadata in enumerate(results["metadatas"][0]):
        if metadata["category"] == category:
            relevant_docs.append(metadata["text"])
    
    return relevant_docs

def answer_question(question, documents):
    """Generate an answer based on retrieved documents."""
    context = "\n".join(documents)
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    raw_response = chain.invoke({"question": question, "context": context})
    
    return raw_response.strip()

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Upload and store a PDF with a category in ChromaDB."""
    if 'file' not in request.files or 'category' not in request.form:
        return jsonify({"error": "File and category are required"}), 400

    file = request.files['file']
    category = request.form['category'].strip().lower()  # Normalize category name

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file.save(tmp.name)
            docs = split_text(load_pdf(tmp.name))
        
        index_docs(docs, category)
        os.remove(tmp.name)

        return jsonify({"message": f"PDF processed and indexed under category '{category}'"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Retrieve relevant documents based on category and answer the question."""
    data = request.get_json()
    question = data.get("question")
    category = data.get("category")

    if not question or not category:
        return jsonify({"error": "Question and category are required"}), 400

    try:
        relevant_docs = retrieve_docs(question, category)
        if not relevant_docs:
            return jsonify({"answer": "No relevant documents found in this category."}), 200
        
        final_response = answer_question(question, relevant_docs)
        return jsonify({"answer": final_response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/categories', methods=['GET'])
def get_categories():
    """Retrieve all unique categories stored in ChromaDB."""
    try:
        results = collection.get()
        categories = set(metadata["category"] for metadata in results["metadatas"])
        return jsonify({"categories": list(categories)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/documents/<category>', methods=['GET'])
def get_documents_by_category(category):
    """Retrieve all documents stored under a specific category."""
    try:
        results = collection.get()
        docs = [
            {"id": results["ids"][i], "text": metadata["text"]}
            for i, metadata in enumerate(results["metadatas"])
            if metadata["category"] == category
        ]
        return jsonify({"documents": docs}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_document', methods=['DELETE'])
def delete_document():
    """Delete a document from ChromaDB by ID."""
    data = request.get_json()
    doc_id = data.get("doc_id")

    if not doc_id:
        return jsonify({"error": "Document ID is required"}), 400

    try:
        collection.delete(ids=[doc_id])
        return jsonify({"message": f"Document {doc_id} deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
