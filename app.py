import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# ---------- FUNCTIONS ----------

def get_pdf_text(pdf_path):
    """Extract text from a single PDF file"""
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
    """Split text into manageable chunks"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


def create_vector_store(text_chunks, api_key):
    """Convert text into embeddings and store in FAISS"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    return FAISS.from_texts(text_chunks, embeddings)


def ask_gemini(vector_store, question, api_key):
    """Ask a question using Gemini model"""
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-8b",
        google_api_key=api_key,
        temperature=0.7
    )

    # Search for most relevant chunks
    docs = vector_store.similarity_search(question, k=3)

    # Run question-answering chain
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain.run(input_documents=docs, question=question)


# ---------- MAIN PROGRAM ----------

def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("❌ Missing GOOGLE_API_KEY in .env file.")
        return

    # Ask for the PDF file
    pdf_path = input("📄 Enter PDF filename (in current folder): ").strip()

    if not os.path.exists(pdf_path):
        print(f"❌ File '{pdf_path}' not found.")
        return

    print("\n📚 Reading and processing PDF...")
    raw_text = get_pdf_text(pdf_path)
    text_chunks = get_text_chunks(raw_text)
    vector_store = create_vector_store(text_chunks, api_key)

    # Ask the user for a question
    question = input("\n💬 Ask your question: ").strip()
    print("\n🤔 Thinking...\n")

    try:
        answer = ask_gemini(vector_store, question, api_key)
        print("✨ Answer:\n")
        
        print(answer)
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
