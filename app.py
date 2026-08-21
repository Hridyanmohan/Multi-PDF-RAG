import os

from flask import Flask, render_template, request, redirect, url_for

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv


# ==========================================
# ENVIRONMENT
# ==========================================

load_dotenv()


# ==========================================
# FLASK
# ==========================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
CHROMA_PATH = "chroma_db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ==========================================
# EMBEDDINGS
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# ==========================================
# LLM
# ==========================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# ==========================================
# INDEX PDFs
# ==========================================

def index_single_pdf(file_path):

    print(f"Processing new PDF: {file_path}")

    # Load PDF
    loader = PyPDFLoader(file_path)

    documents = loader.load()

    print(f"Pages loaded: {len(documents)}")

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    # Existing ChromaDB
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # Add ONLY new chunks
    vector_store.add_documents(chunks)

    print("New PDF embeddings added to ChromaDB.")

    return vector_store

def index_existing_pdfs():

    pdf_files = [
        file
        for file in os.listdir(UPLOAD_FOLDER)
        if file.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("No existing PDFs found.")
        return None

    print(f"Found {len(pdf_files)} existing PDF files.")

    all_documents = []

    for filename in pdf_files:

        file_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        print(f"Loading: {filename}")

        loader = PyPDFLoader(file_path)

        documents = loader.load()

        print(f"  Pages loaded: {len(documents)}")

        all_documents.extend(documents)

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(
        all_documents
    )

    print(f"Total chunks created: {len(chunks)}")

    # Create ChromaDB
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print("Initial embeddings created.")
    print("Documents stored in ChromaDB.")

    return vector_store

def get_vector_store():

    if os.path.exists(CHROMA_PATH):

        print("Existing ChromaDB found.")

        return Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )

    else:

        print("ChromaDB not found.")
        print("Creating initial vector database...")

        return index_existing_pdfs()
# ==========================================
# RAG CHAIN
# ==========================================

def create_rag_chain(vector_store):

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant answering questions
        based only on the provided PDF documents.

        Use the context below to answer the question.

        If the answer cannot be found in the documents,
        say that you don't know.

        Context:
        {context}

        Question:
        {input}

        Answer:
        """
    )

    question_answer_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    rag_chain = create_retrieval_chain(
        retriever,
        question_answer_chain
    )

    return rag_chain


# ==========================================
# HOME
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================
# UPLOAD
# ==========================================

@app.route("/upload", methods=["POST"])
def upload_files():

    files = request.files.getlist("files")

    uploaded_count = 0

    for file in files:

        if file and file.filename.lower().endswith(".pdf"):

            file_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                file.filename
            )

            file.save(file_path)

            print(f"Uploaded: {file.filename}")

            # Process ONLY this PDF
            index_single_pdf(file_path)

            uploaded_count += 1

    print(
        f"{uploaded_count} PDF file(s) uploaded successfully."
    )

    return redirect(url_for("home"))

# ==========================================
# ASK QUESTION
# ==========================================

@app.route("/ask", methods=["POST"])
def ask_question():

    question = request.form.get("question")

    if not question or not question.strip():

        return redirect(url_for("home"))

    print("\nUser Question:")
    print(question)

    # Load existing ChromaDB
    vector_store = get_vector_store()

    # Create RAG chain
    rag_chain = create_rag_chain(
        vector_store
    )

    # Ask question
    response = rag_chain.invoke({
        "input": question
    })

    answer = response["answer"]

    print("\nAI Answer:")
    print(answer)

    return render_template(
        "index.html",
        question=question,
        answer=answer
    )


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)