from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import json
import os
from typing import Annotated, TypedDict, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage , SystemMessage
from langchain_teddynote.graphs import visualize_graph
from langchain_core.runnables import RunnableLambda, RunnableMap, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
import ast

# RAG 관련 모듈 추가
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, List


import os
from dotenv import load_dotenv

## .env 파일 로드
load_dotenv('../.env')


# ModelInfo.json = 모델 관련 정보
model_info_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'ModelInfo.json')
model_info_path = os.path.abspath(model_info_path)

### 모델 관리 클래스
class ModelManager():
    
    def __init__(self, select_models):
        with open(model_info_path , "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        # json 파일에서 읽은 파일 정보 = 모델명/회사/인풋토큰가격/아웃풋토큰가격
        self.total_model_info = loaded_data
        # 선택한 모델들 생성 >> { 모델명 : 모델 ....}
        self.models = { model_name : self.init_Models(model_name) for model_name in select_models if model_name in self.total_model_info.keys()}
        
        # 선택한 모델들 가격정보 >> { 모델명 : {'Input' : .. , 'Output' : .. } ......}
        self.models_info = { model_name : self.total_model_info[model_name] for model_name in self.models.keys()}
    
    # 모델 생성
    def init_Models(self, model_name):
        match self.total_model_info[model_name]['Company']:
            case 'OpenAI':
                return self.init_OpenAI_Model(model_name)
            case 'Anthropic':
                return self.init_Claude_Model(model_name)
            case 'Perplexity':
                return self.init_Perplexity_Model(model_name)      
    
    @staticmethod
    def init_OpenAI_Model(model_name):
        return ChatOpenAI(model=model_name)
    
    @staticmethod
    def init_Claude_Model(model_name):
        return ChatAnthropic(model=model_name)
    
    @staticmethod
    def init_Perplexity_Model(model_name):
        return ChatOpenAI(openai_api_key=os.getenv("PERPLEXITY_API_KEY"),  # PPLX API 키
                          openai_api_base="https://api.perplexity.ai",  # Perplexity API 엔드포인트
                          model_name=model_name,  # 또는 pplx-7b-online, pplx-70b-online 등
                          temperature=0.7)

    @staticmethod
    def init_Local_Model(model_name):
        return ChatOpenAI(openai_api_key=os.getenv("PERPLEXITY_API_KEY"),  # PPLX API 키
                          openai_api_base="https://api.perplexity.ai",  # Perplexity API 엔드포인트
                          model_name=model_name,  # 또는 pplx-7b-online, pplx-70b-online 등
                          temperature=0.7)

if __name__ == "__main__":
    
    model_manager = ModelManager(['gpt-4o-mini' , 'gpt-4o'])
    print(model_manager.models)
    print(model_manager.models_info)
    