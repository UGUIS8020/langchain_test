import langchain
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_community.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
from langchain.agents.agent_toolkits import VectorStoreToolkit, VectorStoreInfo
from langchain.schema import Document

from typing import List
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from langchain.text_splitter import CharacterTextSplitter

langchain.verbose = True

load_dotenv()

# langsmithを使うためのコード
openai_api_key = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGCHAIN_PROJECT'] = "LangSmith-test"



def create_index(force_reload: bool = False, add_new_data: bool = False) -> VectorStoreIndexWrapper:
    """
    Create, load, or update a VectorStoreIndexWrapper.
    
    Args:
        force_reload (bool): If True, recreate the index even if it already exists.
        add_new_data (bool): If True, add new data to the existing index.
    
    Returns:
        VectorStoreIndexWrapper: The created, loaded, or updated index.
    """
    persist_directory = "chroma_db"
    
    def load_and_process_documents() -> List[Document]:
        loader = DirectoryLoader("text/", glob="**/*.txt")
        documents = loader.load()
        splitter = CharacterTextSplitter(separator="。", chunk_size=100, chunk_overlap=0)
        return splitter.split_documents(documents)

    # Check if the index already exists
    if os.path.exists(persist_directory) and not force_reload:
        try:
            print("Loading existing index...")
            embedding = OpenAIEmbeddings(model="text-embedding-3-large")
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)
            
            if add_new_data:
                print("Adding new data to existing index...")
                new_documents = load_and_process_documents()
                vectorstore.add_documents(new_documents)
                print(f"Added {len(new_documents)} new document chunks to the index.")
            
            return VectorStoreIndexWrapper(vectorstore=vectorstore)
        except Exception as e:
            print(f"Error loading existing index: {e}")
            print("Proceeding to create a new index...")
    
    # Create a new index
    if force_reload or not os.path.exists(persist_directory):
        try:
            print("Creating new index...")
            split_docs = load_and_process_documents()
            embedding = OpenAIEmbeddings(model="text-embedding-3-large")
            vectorstore = Chroma(embedding_function=embedding, persist_directory=persist_directory)
            vectorstore.add_documents(split_docs)
            print(f"Created new index with {len(split_docs)} document chunks.")
            return VectorStoreIndexWrapper(vectorstore=vectorstore)
        except Exception as e:
            print(f"Error creating new index: {e}")
            raise
    
    # If we reach here, it means the index exists and we're not forcing a reload or adding new data
    print("Using existing index without modifications.")
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    return VectorStoreIndexWrapper(vectorstore=vectorstore)

# インデックスの作成、読み込み、または更新
# 既存のインデックスを使用する場合
index = create_index()  

# 新しいデータを追加する場合
# index = create_index(add_new_data=True) 

# インデックスを強制的に再作成する場合
# index = create_index(force_reload=True) 

def create_tools(index: VectorStoreIndexWrapper, llm) ->List[BaseTool]:
    
    vectorstore_info = VectorStoreInfo(
        name="test_text_code",
        description="A collection of text documents for testing purposes.",
        vectorstore=index.vectorstore,        
    )
    
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)
    return toolkit.get_tools()


def chat(message: str, history: ChatMessageHistory, index: VectorStoreIndexWrapper) -> str:
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    tools = create_tools(index, llm)
    memory = ConversationBufferMemory(chat_memory=history, memory_key="chat_history", return_messages=True)
    agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, memory=memory)
    
   
    return agent_chain.run(input=message)

