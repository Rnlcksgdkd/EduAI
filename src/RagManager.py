
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


from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import json
import os
import random
from typing import Annotated, TypedDict, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage , SystemMessage
from langchain_teddynote.graphs import visualize_graph
from langchain_core.runnables import RunnableLambda, RunnableMap, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
import ast
from IPython.display import Image, display


# RAG 관련 모듈 추가
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, List
from datetime import datetime
from langchain_core.output_parsers import PydanticOutputParser
from typing import Literal, get_args

# src 내 모듈 import
import ModelManager
import PromptTemplate
import RagManager
import Structure

## RAG 문서 경로
docs_path = os.path.join(os.path.dirname(__file__), '..', 'docs')
docs_path = os.path.abspath(docs_path)


## 문서 종류에 따라 조건분기 함수 
def docu_routing(state):
    
    ''' 문서 종류에 따라 조건분기 함수 '''
    
    file_path = state.get('file_path' , None)       # 파일 경로
    
    if file_path == None: return 'None'
    
    file_ext = file_path.split('.')[-1]             # 파일 확장자
    
    # pdf , txt , pptx 파일만 처리
    if file_ext in ['pdf' , 'txt' , 'pptx']:
        return file_ext
    
    # 다른 파일 형식은 None 반환 (RAG 적용 x)
    return 'None'


# pdf 파일 로더
def load_pdf_docu(state):
    
    """ PDF 파일 로더 """
    
    file_path = state.get('file_path')
    return {'input_docu' : load_pdf_docu_path(file_path)}


def load_pdf_docu_path(file_path):
    
    """ PDF 파일 로더 """
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    return documents


# txt 파일 로더 >> 
def load_txt_docu(state):
    
    """ TXT 파일 로더 """
    
    file_path = state.get('file_path')
    
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        docu = loader.load() 
        return {'input_docu' : loader.load() }
    
    except UnicodeDecodeError:
        loader = TextLoader(file_path, encoding='cp949')  # 한글 윈도우 인코딩
        return {'input_docu' : loader.load() }


# pptx 파일 로더
def load_pptx_docu(state):
    
    """ PPTX 파일 로더 """
    pass


def print_pdf_docs(docs):
    
    for page in range(docs[0].metadata['total_pages']):
        print(f"--- 페이지 {page + 1}")
        print(docs[page].page_content)


class RAGManager:
    
    def __init__(self, docs_dir=docs_path):
        
        self.docs_dir = docs_dir
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        
    def load_and_process_documents(self , doc):
        
        """문서를 로드하고 벡터화하여 저장"""
        
        # 텍스트 분할기
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        # Chunk 분할
        chunks = text_splitter.split_documents(doc)
        
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


## docs 를 문자열로 반환
def docs2str(docs):
    
    doc_str = ""
    for i , doc in enumerate(docs):
        doc_str += f"------ 페이지 {i + 1} -------\n"
        doc_str += doc.page_content + "\n\n"

    return doc_str

def rag_2(input_docu , query):
    
    rag_manager = RAGManager()
    rag_manager.load_and_process_documents(input_docu)
    relevant_docs = rag_manager.get_relevant_documents(query)
    
    print("="*100)
    print('RAG 쿼리 결과 : ')
    for doc in relevant_docs:
        print(doc.page_content)
    
    context = rag_manager.get_context_str(relevant_docs)
    
    return context


def rag_query(state):
    
    context = ""
    input_docu = state['input_docu']
    query = state.get('query' , '')
    
    match state['rag_option']:
        case 0:
            pass
        case 1:
            context = docs2str(input_docu)
        case 2:
            context = rag_2(input_docu , query)
        case 3:
            pass
     
    return {'context' : context}
  



def rag_module():
    
    # StateGraph 생성
    builder = StateGraph(Structure.ragState)

    ## 노드 설정 ##
    builder.add_node( "Load_PDF" , load_pdf_docu)        
    builder.add_node( 'Load_TXT' , load_txt_docu)      
    builder.add_node( 'Load_PPT' , load_pptx_docu)      
    builder.add_node( 'RAG' , rag_query)           
         

    # 엣지 설정 ##
    builder.add_edge('Load_PDF', 'RAG')
    builder.add_edge('Load_TXT', 'RAG')
    builder.add_edge('Load_PPT', 'RAG')
    builder.add_edge('RAG', END)
    
    ## 조건부 엣지 설정 ##
    builder.add_conditional_edges( START, docu_routing ,
        { 'pdf' : 'Load_PDF',
           'txt' : 'Load_TXT',
           'pptx' : 'Load_PPT' , 'None' : END }
    )


    # ## 조건부 엣지 설정 ##
    # builder.add_conditional_edges( START, docu_routing ,
    #     { 'pdf' : END}
    # )
    
    # 그래프 컴파일
    app = builder.compile(checkpointer = MemorySaver())
    
    # img_txt = app.get_graph().draw_mermaid()
    # print(img_txt)
    
    # with open("RAG_Graph.png", "wb") as f:
    #     f.write(img.data)
        
    return app


def exec_rag(input_dict):
    
    RAG_graph = rag_module()
    
    # Config 설정
    config = RunnableConfig(
        recursion_limit=10,  # 최대 10개의 노드까지 방문. 그 이상은 RecursionError 발생
        configurable={"thread_id": "7"},  # 스레드 ID 설정
        tags=["my-tag"],  # Tag
    )

    event_list = []
    
    # Graph 실행
    for event in RAG_graph.stream(input =  input_dict, config=config):
        event_list.append(event)
        
    return event_list


if __name__ == "__main__":
    
    ## .env 파일 로드
    load_dotenv('../.env')


    ### 1). TXT 로더 테스트 ###
    txt_file_path = '../docs/빅데이터분석기사.txt'
    # test_docu = load_txt_docu(test_file_path)
    # print(test_docu[0].page_content)
    
    
    ### 2). PDF 로더 테스트 ###
    pdf_file_path_1 = '../docs/2024 내일은 빅데이터분석기사 필기 핵심 요약집.pdf'
    pdf_file_path_2 = '../docs/온디바이스 AI 기술동향 및 발전방향.pdf'
    
    docs = load_pdf_docu_path(pdf_file_path_2)
    print_pdf_docs(docs)
    
    
    ### 3). 텍스트 통채로 프롬프트에 전달 (RAG 옵션 1) ###
    input_dict = {'file_path' : pdf_file_path_2  , 'rag_option' : 1 , 'query' : '데이터 전처리'}
    exec_rag(input_dict)



    ### 4). RAG 옵션2 테스트 ###
    

    
    
    # # RAG 매니저 인스턴스 생성
    # rag_manager = RAGManager()
    
    # # 문서 로드 및 처리
    # num_docs = rag_manager.load_and_process_documents()
    # print(f"Loaded {num_docs} documents.")
    
    # # 쿼리 검색
    # query = "ADsP 통계"
    # docs = rag_manager.get_relevant_documents(query)
    
    # # 문맥 문자열 생성
    # context_str = rag_manager.get_context_str(docs)
    # print("Context String:")
    # print(context_str)  
    
    
