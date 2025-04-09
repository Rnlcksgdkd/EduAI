
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import json
import os
from typing import Annotated, TypedDict, List, Literal
from dotenv import load_dotenv

# RAG 관련 모듈 추가
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Annotated, List


## RAG 문서 경로
docs_path = os.path.join(os.path.dirname(__file__), '..', 'docs')
docs_path = os.path.abspath(docs_path)


### RAG 관련 클래스 및 함수 추가 ###
class RAGManager:
    
    def __init__(self, docs_dir=docs_path):
        self.docs_dir = docs_dir
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        
    def load_and_process_documents(self):
        
        """문서를 로드하고 벡터화하여 저장"""
        
        # 문서 로더 설정
        loaders = []
        
        # PDF 파일 로더 추가
        pdf_loader = DirectoryLoader(self.docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
        loaders.append(pdf_loader)
        
        # 텍스트 파일 로더 추가
        text_loader = DirectoryLoader(self.docs_dir, glob="**/*.txt", loader_cls=TextLoader)
        loaders.append(text_loader)
        
        # 문서 로드
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        
        # 텍스트 분할기
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        # Chunk 분할
        chunks = text_splitter.split_documents(docs)
        
        # 벡터 저장소 생성
        self.vector_store = Chroma.from_documents(documents=chunks, embedding=self.embeddings)
        
        return len(chunks)

    # 검색 
    def get_relevant_documents(self, query, k=3):
        
        """쿼리에 관련된 문서 검색"""
        if not self.vector_store:
            raise ValueError("벡터 저장소가 초기화되지 않았습니다. load_and_process_documents()를 먼저 호출하세요.")
        
        docs = self.vector_store.similarity_search(query, k=k)
        return docs
    
    def get_context_str(self, docs):
        """검색된 문서를 문맥 문자열로 변환"""
        context = "\n\n".join([doc.page_content for doc in docs])
        return context

if __name__ == "__main__":
    
    ## .env 파일 로드
    load_dotenv('../.env')
    
    # RAG 매니저 인스턴스 생성
    rag_manager = RAGManager()
    
    # 문서 로드 및 처리
    num_docs = rag_manager.load_and_process_documents()
    print(f"Loaded {num_docs} documents.")
    
    # 쿼리 검색
    query = "ADsP 통계"
    docs = rag_manager.get_relevant_documents(query)
    
    # 문맥 문자열 생성
    context_str = rag_manager.get_context_str(docs)
    print("Context String:")
    print(context_str)  