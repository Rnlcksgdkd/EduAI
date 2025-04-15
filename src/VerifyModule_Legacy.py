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
from dotenv import load_dotenv


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

def func(state):
    
    if 1 == "Search Grounding" :
        return "Search Grounding" 
    else:
        return "Web Check" 


def routing1(state):
    return {
        
        
    }


## 문제 생성 모듈 (Graph) ##
def validate_module():

    # StateGraph 생성
    builder = StateGraph(Structure.State)

    # 검증에 필요한 RAG 모듈 생성
    rag_validate = RagManager.rag_module()

    ## 노드 설정 ##
    builder.add_node("RAG", func)
    
    builder.add_node("Search Grounding", func)
    builder.add_node("Web Check", func)
    
    builder.add_node("Problem Modify", func)
    
    builder.add_node("Critic Agents", func)
    builder.add_node("Phrasing Modify", func)
    
    builder.add_node("Solve Agents", func)

    builder.set_entry_point("RAG")
    builder.add_edge("Search Grounding", "Problem Modify")
    builder.add_edge("Web Check", "Problem Modify")
    builder.add_edge("Problem Modify", "Critic Agents")
    builder.add_edge("Critic Agents", "Problem Modify")
    builder.add_edge("Problem Modify", "Phrasing Modify")
    builder.add_edge("Problem Modify", "Solve Agents")
    

    ## 조건부 엣지 설정 ##
    builder.add_conditional_edges( "RAG", routing1 ,
        { "Search Grounding"  : "Search Grounding"  ,   "Web Check" : "Web Check" }
    )

    # 그래프 컴파일
    app = builder.compile(checkpointer = MemorySaver())
    img_txt = app.get_graph().draw_mermaid()
    print(img_txt)
    
    
    
    










if __name__ == '__main__':
    app = validate_module()