import os
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TaiwanLawRAG:
    def __init__(self):
        # List of important laws with their URLs
        self.base_urls = [
            "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=B0000001",  
            "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=C0000001",  
            "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=C0010001",  
            "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=N0030001",  
            "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=N0060009",
            "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=N0060001",
            "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=B0010001",
            "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=J0080001",  
            "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=K0040012",  
            "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=A0030055",
            "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=N0060027",
            "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=A0000001",
            "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=N0060002",
            "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=A0030057",
            "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=K0040013",
            "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=G0340003",
            "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=G0380131",
            "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=D0060001",
            "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=N0030014",
            "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=N0060014",
        ]
        self.chat_history = []
        self.setup_rag()

    def setup_rag(self):
        # 1. Load and process the data from multiple URLs
        documents = []
        for url in self.base_urls:
            try:
                loader = WebBaseLoader(web_paths=[url])
                docs = loader.load()
                documents.extend(docs)
                print(f"Successfully loaded: {url}")
            except Exception as e:
                print(f"Error loading {url}: {str(e)}")

        if not documents:
            raise ValueError("No documents were successfully loaded")

        # 2. Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        splits = text_splitter.split_documents(documents)
        print(f"Created {len(splits)} document chunks")

        # 3. Create embeddings and vector store
        embedding = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory="./taiwan_law_db"
        )

        # 4. Create the conversational chain
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0,
            streaming=True,
            verbose=False
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}
            ),
            return_source_documents=True,
            verbose=True
        )

    def query(self, question: str) -> str:
        """
        Query the RAG system with a question
        """
        result = self.qa_chain({
            "question": question,
            "chat_history": self.chat_history
        })
        
        # Update chat history
        self.chat_history.append((question, result["answer"]))
        
        return result["answer"]

    def clear_history(self):
        """
        Clear the chat history
        """
        self.chat_history = []

# Replace the hardcoded testing section with this interactive version:
def main():
    # Initialize the system
    print("Initializing Taiwan Law RAG system...")
    law_rag = TaiwanLawRAG()
    print("\nSystem ready! Type 'quit' or 'exit' to end the session, 'clear' to clear chat history.")
    
    while True:
        # Get user input
        question = input("\nPlease enter your legal question: ").strip()
        
        # Check for exit commands
        if question.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        # Check for clear history command
        if question.lower() == 'clear':
            law_rag.clear_history()
            print("Chat history cleared!")
            continue
        
        # Skip empty questions
        if not question:
            print("Please enter a valid question.")
            continue
            
        try:
            print("\nSearching for answer...")
            response = law_rag.query(question)
            print(f"\nAnswer: {response}")
            print("\n" + "-"*50)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()