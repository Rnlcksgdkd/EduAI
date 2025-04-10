'''생성한 문제의 논리적 오류를 검증하는 모듈 (서브그래프)'''
'''생성한 문제를 불러와서 프롬프트 만들고 LLM 에게 요청'''
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

# src 내 모듈 import
from ModelManager import ModelManager
import PromptTemplate
import RagManager
import Structure
from dotenv import load_dotenv


NODE_LOGIC_VERIFY = 'Logic Check'


class State(TypedDict):
    
    ## 입력 ##
    models : dict
    models_info : dict
    input_questions : dict

    logic_check : dict
    logic_pass : dict
    logic_fail : dict


# 4지선다형 문제들 로직 체크
def MultipleChoiceQuestion_logic_check(model , question):

    print("뭐임?")
    Logic_Check_promptMessage = ChatPromptTemplate.from_messages([
    ("system", PromptTemplate.Logic_Check_systemTemplate),
    ("human", PromptTemplate.Logic_Check_questionTemplate)
    ])

    choices = ["choice_1" , "choice_2" , "choice_3" , "choice_4" , 'answer']

    parser = PydanticOutputParser(pydantic_object=Structure.LogicCheck)

    one_question_check = []
    for choice in choices:

        prompt = Logic_Check_promptMessage.format_messages(
            question = question['question'] , answer = question[choice] , format = parser.get_format_instructions())
        response = model.invoke(prompt)
        print("="*100)
        print(response)
        print("="*100)

        response_content = json.loads(response.content)
        is_error = response_content['is_error']
        is_error_descript = response_content['is_error_descript']
        # is_error_prob = response_content['is_error_prob']


        print("="*100)
        print(question['question'], "  >>>  " , question[choice])
        print(f"사실 여부 : {is_error}")
        print(f"로직 체크 결과 : {is_error_descript}")
        print("="*100)

        # response.content['is_confused']
        one_question_check.append(is_error)
        
    return one_question_check



def logic_verify(state):

    
    total_question_logic_check = {}

    for model_name in state['models'].keys():

        all_questions_logic_check = []
        question_lst = state['input_questions']
        
    
        for question in question_lst:
            print(model_name , question)
            q_logic_check = MultipleChoiceQuestion_logic_check(state['models'][model_name] , question)
            
            all_questions_logic_check.append(q_logic_check)
        
        total_question_logic_check[model_name] = all_questions_logic_check

    print(total_question_logic_check)
    Flag = True

    return {'logic_check' : total_question_logic_check}



def logicGraph():
    builder = StateGraph(State)
    builder.add_node( NODE_LOGIC_VERIFY, logic_verify)
    builder.set_entry_point(NODE_LOGIC_VERIFY)
    builder.add_edge(NODE_LOGIC_VERIFY, END)
    app = builder.compile(checkpointer = MemorySaver())
    
    return app



import setting


if __name__ == "__main__":

    ## .env 파일 로드
    load_dotenv('./.env')
        
        
    
    ##################### 사용자 입력 #####################
    model_manager = ModelManager(['gpt-4o-mini'])
    input_file_path = setting.generate_save_path + '/AI__20250410_1038.json'
    ######################################################
    

    with open(input_file_path, 'r', encoding='utf-8') as f:
        input_questions = json.load(f)

    # 입력 생성
    input_dict = {
        'models': model_manager.models , 
        'models_info' : model_manager.models_info , 
        'input_questions' : input_questions
    }

    print(model_manager.models)

    # Config 설정
    config = RunnableConfig( recursion_limit=10 , configurable={"thread_id": "7"} )

    # Graph 생성
    app = logicGraph()
    
    # Event 저장 리스트
    event_list = []
    
    # Graph 실행
    for event in app.stream(input=input_dict, config=config):
        event_list.append(event)