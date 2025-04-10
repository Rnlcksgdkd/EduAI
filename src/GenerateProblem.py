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
from langgraph.graph.message import add_messages


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
from ModelManager import ModelManager
import PromptTemplate
import RagManager
import Structure
import setting

'''문제 생성 모듈'''



########################
### 노드 이름 정의 ###
########################
NODE_VERIFY = 'Node_Verify'
NODE_GENERATE = 'Node_Generate'
NODE_PROMPT = 'Node_Prompt'
NODE_RAG = 'Node_RAG'
NODE_LOGIC_VERIFY = 'Node_Logic_Verify'
NODE_PROPER_VERIFY = 'Node_Proper_Verify'
NODE_SAVE_RESULT = 'Node_Save_Result'


# def decide_attr_random(attr):
#     attr_values = get_args( Structure.MultiChoiceQuestion.__annotations__[attr])
#     return random.choice(attr_values)


## 1). RAG 컨텍스트 검색 노드 ##
def retrieve_rag(state , rag_module):
    
    """ RAG 컨텍스트 검색 노드 """

    # State 값 로드
    input_file_path = state.get('input_file_path')  # 입력 레퍼런스 문서 경로
    rag_option = state.get('rag_option')            # RAG 사용 옵션 = 0 : 사용안함 , 1: 입력문서 통채로 전달 , 2: 요청쿼리로 검색
    topic = state.get('topic' , "전체범위")          # RAG 검색 쿼리 = 없다면 전체 범위에 대해 문제 생성
    
    # RAG 사용 안하는 경우는 context를 비워서 반환
    if rag_option == 0 : return {'context' : ""}
    
    # RAG - 서브그래프(src/RagManager.py) 호출
    response = rag_module.invoke({  'file_path' : input_file_path  , 'rag_option' : rag_option , 'query' : topic })
    
    
    print(state['log_messages'])
    
    # RAG 서브그래프 응답 반환
    return {"context": response["context"]}


    
## 2). 프롬프트 노드 ##
def problem_generate_prompt(state):
    
    """ 문제 생성 프롬프트 노드 """
    
    # State 값 로드
    input_file_path = state.get('input_file_path')  # 입력 레퍼런스 문서 경로
    file_name = input_file_path.split('/')[-1].split('.')[0] if input_file_path != None else None
    print("## 테스트 1 : " , file_name)
    title = state.get("title", file_name )  # 시험 이름 (없는 경우는 입력레퍼런스 파일이름)
    topic = state.get("topic", "범위 전체")          # 범위 전체
    num_question = state.get('num_question')        # 생성할 문제 수
    context = state.get("context")                  # RAG에서 검색된 문맥 정보

    # 생성할 문제 형식을 파서로 정의 (객관식 - 4지선다)
    parser = PydanticOutputParser(pydantic_object=Structure.MultiChoiceQuestion)
    
    # 문제 생성에 사용할 프롬프트 생성
    prompt_messages = PromptTemplate.standard_prompt_template.format_messages(
        title=title,
        topic = topic,
        context = context,
        num_question=num_question,
        format=parser.get_format_instructions()
    )
        
    # State에 prompt_message 추가
    return {"messages" : prompt_messages , "prompt_message": prompt_messages , 'graph_flow' : (state.get('graph_flow', []) + [NODE_PROMPT]) , 'node_name' : NODE_PROMPT }

## ✅ 3. LLM 모델 응답 생성 노드 (병렬 처리) ##
generate_problem = RunnableLambda(lambda state : build_parallel_model_map(state.get('models').keys()))

# 병렬 처리를 위한 RunnableMap 리턴 함수 #
def build_parallel_model_map(model_names):
    
    """ 병렬 처리로 모델 응답 생성 """
    
    return RunnableMap({
    "messages": lambda state: state["messages"],
    "model_response": RunnableMap({
        model_name: RunnableLambda(lambda state, m=model_name: generate_response(state, m)) for model_name in model_names
    }),
    "title": lambda state: state["title"],
    "num_question": lambda state: state["num_question"],
    "context": lambda state: state["context"],
    'graph_flow' : lambda state : (state.get('graph_flow', []) + [NODE_GENERATE]) ,
    'node_name' : lambda state : 'Node_Generate'
})


# 개별 모델 응답 함수
def generate_response(state, model_name):

    # 모델 로드
    model = state.get('models')[model_name]
    
    # LLM 응답 생성
    response = model.invoke(state["messages"])
    
    # 인풋/아웃풋 토큰 가격 계산
    model_input_tokens = response.usage_metadata['input_tokens']
    model_output_tokens = response.usage_metadata['output_tokens']
    response_price = state.get('models_info')[model_name]['Input']*model_input_tokens + state.get('models_info')[model_name]['Output']*model_output_tokens

    # 문제 객체화
    json_str = response.content
    str_question_lst = ast.literal_eval(json_str)

    return {'content': response.content, 'cost': response_price , 'questions' : str_question_lst }


# 생성한 문제 저장 및 출력
def save_result(state):

    state['graph_flow'] += [NODE_SAVE_RESULT]

    for model_name in state['model_response']:
        
        print(f"✅ 답변 모델: {model_name}")
        print(f"✅ 토큰 발생 비용: {state['model_response'][model_name]['cost']}")

        questions = state['model_response'][model_name]['questions']
        
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M")
        
        file_name = f"{setting.generate_save_path}/{state['title']}_{state['topic']}_{datetime_str}.json"
        
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=4)
       
        print(f"✅ JSON 파일 저장 완료: {file_name}")    
        return {'node_name' : NODE_SAVE_RESULT} 




    
## 문제 생성 모듈 (Graph) ##
def generate_problem_module():

    """ 문제 생성 모듈
    START >> NODE_RAG >> NODE_PROMPT >> NODE_GENERATE >> NODE_SAVE_RESULT >> END
    """

    # StateGraph 생성
    builder = StateGraph(Structure.State)

    # 문제 생성에 필요한 RAG 모듈 생성
    rag_graph = RagManager.rag_module()

    ## 노드 설정 ##
    builder.add_node( NODE_RAG, RunnableLambda(lambda state: retrieve_rag(state , rag_graph)))       
    builder.add_node( NODE_PROMPT , problem_generate_prompt)        
    builder.add_node( NODE_GENERATE , generate_problem)      
    builder.add_node( NODE_SAVE_RESULT , save_result)           

    ## 엣지 설정 ##
    builder.add_edge(START, NODE_RAG)
    builder.add_edge(NODE_RAG, NODE_PROMPT)
    builder.add_edge(NODE_PROMPT, NODE_GENERATE)
    builder.add_edge(NODE_GENERATE, NODE_SAVE_RESULT)
    builder.add_edge(NODE_SAVE_RESULT, END)
    
    # 그래프 컴파일
    app = builder.compile(checkpointer = MemorySaver())
    
    return app



if __name__ == "__main__":
    
    
    ## .env 파일 로드
    load_dotenv('./.env')
        
    '빅데이터분석기사.txt'
    "온디바이스 AI 기술동향 및 발전방향"
    "2024 내일은 빅데이터분석기사 필기 핵심 요약집"
    
    
    ##################### 사용자 입력 #####################
    model_manager = ModelManager(['gpt-4o-mini'])
    input_file_path = '../docs/빅데이터분석기사.txt'
    title = "AI"
    topic = ""
    num_question = 5
    rag_option = 1
    ######################################################
    
    
    # 입력 생성
    input_dict = {
    "models" : model_manager.models,
    "models_info" : model_manager.models_info,
    "input_file_path" : input_file_path , 
    "title" :  title,
    "topic" : topic,
    "num_question": num_question,
    'rag_option' : rag_option
    }

    # Config 설정
    config = RunnableConfig(
        recursion_limit=10,  # 최대 10개의 노드까지 방문. 그 이상은 RecursionError 발생
        configurable={"thread_id": "7"},  # 스레드 ID 설정
        tags=["my-tag"],  # Tag
    )

    # Graph 생성
    app = generate_problem_module()
    

    # # Event 저장 리스트
    # test_list = []
    event_list = []
    
    # Graph 실행
    for event in app.stream(input=input_dict, config=config):
        event_list.append(event)