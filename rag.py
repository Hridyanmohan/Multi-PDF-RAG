import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
print("Groq API Key loaded:", bool(os.getenv("GROQ_API_KEY")))
UPLOAD_FOLDER = "uploads"


def load_pdfs():

    documents = []

    # Get all PDF files from uploads folder
    pdf_files = [
        file for file in os.listdir(UPLOAD_FOLDER)
        if file.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("No PDF files found in uploads folder.")
        return []

    print(f"Found {len(pdf_files)} PDF files.")

    # Load each PDF using LangChain
    for filename in pdf_files:

        pdf_path = os.path.join(UPLOAD_FOLDER, filename)

        print(f"Loading: {filename}")

        loader = PyPDFLoader(pdf_path)

        pages = loader.load()

        print(f"  Pages loaded: {len(pages)}")

        documents.extend(pages)

    print("\n================================")
    print(f"Total PDF files: {len(pdf_files)}")
    print(f"Total pages loaded: {len(documents)}")
    print("================================")

    return documents
def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print("\n================================")
    print(f"Total chunks created: {len(chunks)}")
    print("================================")

    return chunks
def create_vector_database(chunks):

    print("\nCreating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print("Embeddings created successfully.")
    print("Documents stored in ChromaDB.")

    return vector_store

def create_retriever(vector_store):

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever


def test_retriever(retriever):

    query = input("\nEnter your question: ")

    documents = retriever.invoke(query)

    print("\n--- Retrieved Documents ---")

    for i, doc in enumerate(documents, start=1):

        print(f"\nChunk {i}")
        print("-" * 50)

        print(doc.page_content[:1000])

        print("\nMetadata:")
        print(doc.metadata)
def create_llm():

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

    return llm

def create_prompt():

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant answering questions based only
        on the provided documents.

        If the answer cannot be found in the documents,
        say that you don't know.

        Context:
        {context}

        Question:
        {input}

        Answer:
        """
    )

    return prompt   

def create_rag_chain(retriever, llm):

    prompt = create_prompt()

    question_answer_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    rag_chain = create_retrieval_chain(
        retriever,
        question_answer_chain
    )

    return rag_chain     
        
if __name__ == "__main__":

    documents = load_pdfs()

    chunks = split_documents(documents)

    vector_store = create_vector_database(chunks)

    retriever = create_retriever(vector_store)

    llm = create_llm()

    rag_chain = create_rag_chain(
        retriever,
        llm
    )

    question = input("\nEnter your question: ")

    response = rag_chain.invoke({
        "input": question
    })

    print("\n--- AI Answer ---")
    print(response["answer"])