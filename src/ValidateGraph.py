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
from SolverAgent import SolverAgent
from CriticAgent import CriticAgent


class ValidateState(TypedDict):
    
    questionSet_path : str
    messages: dict

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        questionSet = json.load(f)
    return questionSet

def solver(state):
    
    
    # 1. 문제 세트 로드 >> Solver 에게 전달할 값들만 추출
    questionSet = load_json_file(state['questionSet_path'])
    state['messages']['Question'] = questionSet
    
    need_key = ['question' , 'choice_1' , 'choice_2' , 'choice_3' , 'choice_4']
    convert_questionSet = [ { key : value for key , value in q.items() if key in need_key} for q in questionSet ]

    # 2. SolverAgent 생성 및 실행
    solver_agent = SolverAgent()
    runner = solver_agent.run_batch()
    results = [ runner.invoke({'question_info' : q })['solver_answer'].dict() for q in convert_questionSet]
    state['messages']['Solver'] = results
    
    return {}


def critic(state):
    
    question_batch = state['messages']['Question']
    solver_batch = state['messages']['Solver']

    # Example usage
    critic_agent = CriticAgent()
    runner = critic_agent.run_batch()
    
    results = [ runner.invoke({'question_info' : question_batch[i]  , 'solver_answer' : solver_batch[i]})['critic_answer'].dict() for i in range(len(question_batch))]
    state['messages']['Critic'] = results

    return {}
    

def refine(state):
    
    return {}

def save_result(state):
    
    feedback_path = "../feedback/" + state['questionSet_path'].split('/')[-1]
    with open(feedback_path, 'w', encoding='utf-8') as f:
        json.dump(state['messages'] , f, ensure_ascii=False, indent=4)

    
    print("============22==========")
    print(state['messages'])
    

    return {}

## 문제 생성 모듈 (Graph) ##
def validate_module():

    # StateGraph 생성
    builder = StateGraph(ValidateState)

    # 검증에 필요한 RAG 모듈 생성
    rag_validate = RagManager.rag_module()

    ## 노드 설정 ##
    builder.add_node("Solver", solver)
    builder.add_node("Critic", critic)
    builder.add_node("Refine", refine)
    builder.add_node("SaveResult", save_result)
    

    builder.set_entry_point("Solver")
    builder.add_edge("Solver", "Critic")
    builder.add_edge("Critic", "Refine")
    builder.add_edge("Refine", "SaveResult")
    
    
    # 그래프 컴파일
    app = builder.compile()
    # img_txt = app.get_graph().draw_mermaid()
    # print(img_txt)
    
    return app
    
    

if __name__ == "__main__":
    
    
    ########## 테스트 입력  ##########
    questionSet_path = '../generate/AI__20250415_1740.json'
    ################################


    ## 1). 생성한 문제 로드 테스트
    # with open(questionSet_path, 'r', encoding='utf-8') as f:
    #     questionSet = json.load(f)
    # print(questionSet)

    ## 2). 문제 검증 모듈 테스트
    app = validate_module()
    app.invoke( {'questionSet_path' : questionSet_path , 'messages' : {}} )


