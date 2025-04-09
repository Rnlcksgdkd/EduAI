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



def decide_attr_random(attr):
    attr_values = get_args( Structure.MultiChoiceQuestion.__annotations__[attr])
    return random.choice(attr_values)

    
# ✅ 1). RAG 컨텍스트 검색 노드
def retrieve_context(state):
    
    # """RAG를 사용하여 관련 문서를 검색하고 문맥을 추가"""
    # if state.get("use_rag", False):
    return {"context": "", "use_rag": False ,
            'node_name' : NODE_RAG ,  'graph_flow' : (state.get('graph_flow', []) + [NODE_RAG]) }
    
    # 검색 쿼리 생성
    query = f"{state['exam_name']} 시험 {difficulty} 난이도"
    
    # # RAG 매니저를 통해 관련 문서 검색
    # rag_manager = RagManager.RAGManager()
    
    # try:
    #     # 이미 벡터 저장소가 초기화되었다고 가정
    #     docs = rag_manager.get_relevant_documents(query)
    #     context = rag_manager.get_context_str(docs)
    # except ValueError:
    #     # 벡터 저장소가 초기화되지 않은 경우 초기화 후 검색
    #     rag_manager.load_and_process_documents()
    #     docs = rag_manager.get_relevant_documents(query)
    #     context = rag_manager.get_context_str(docs)
    
    # return {"context": context, "difficulty" : difficulty ,
    #         "use_rag": True , 'graph_flow' : (state.get('graph_flow', []) + [NODE_RAG]) , 'node_name' : NODE_RAG}



# ✅ 2). 프롬프트 노드
def prompt_node(state):
    
    
    print("?"*100)
    
    """RAG 여부에 따라 다른 프롬프트 템플릿 사용"""
    # 필요한 변수 추출
    exam_name = state.get("exam_name", "ADsP")
    topic = state.get("topic", "범위 전체")
    num_question = state.get('num_question', 3)
    difficulty = state.get("difficulty", "중")
    use_rag = state.get("use_rag", False)
    context = state.get("context", "")

    parser = PydanticOutputParser(pydantic_object=Structure.MultiChoiceQuestion)
    

    # RAG 사용 여부에 따라 프롬프트 템플릿 선택 (추후)
    messages = PromptTemplate.standard_prompt_template.format_messages(
        exam_name=exam_name,
        topic = topic,
        difficulty = difficulty,
        num_question=num_question,
        format=parser.get_format_instructions()
    )
        
    
    # 메시지를 state에 추가
    return {"messages" : messages , "prompt_message": messages , 'graph_flow' : (state.get('graph_flow', []) + [NODE_PROMPT]) , 'node_name' : NODE_PROMPT }


def build_parallel_model_map(model_names):
    
    return RunnableMap({
    "messages": lambda state: state["messages"],
    "model_response": RunnableMap({
        model_name: RunnableLambda(lambda state, m=model_name: generate_response(state, m)) for model_name in model_names
    }),
    "exam_name": lambda state: state["exam_name"],
    "difficulty": lambda state: state["difficulty"],
    "num_question": lambda state: state["num_question"],
    "use_rag": lambda state: state["use_rag"],
    "context": lambda state: state["context"],
    'graph_flow' : lambda state : (state.get('graph_flow', []) + [NODE_GENERATE]) ,
    'node_name' : lambda state : 'Node_Generate'
})


# ✅ 3. LLM 모델 응답 생성 노드 (병렬 처리)
parallel_models = RunnableLambda(lambda state : build_parallel_model_map(state.get('models').keys()))


# 모델별 응답 생성 함수
def generate_response(state, model_name):

    # 모델 로드
    model = state.get('models')[model_name]
    
    # LLM 응답 생성
    response = model.invoke(state["messages"])
    
    print("##### 테스트 ####")
    print(response)
    
    # 인풋/아웃풋 토큰 가격 계산
    model_input_tokens = response.usage_metadata['input_tokens']
    model_output_tokens = response.usage_metadata['output_tokens']
    response_price = state.get('models_info')[model_name]['Input']*model_input_tokens + state.get('models_info')[model_name]['Output']*model_output_tokens

    # 문제 객체화
    json_str = response.content
    str_question_lst = ast.literal_eval(json_str)

    # question_lst = [MultiChoiceQuestion(**json) for json in str_question_lst]

    print(str_question_lst)
    
    return {'content': response.content, 'cost': response_price , 'questions' : str_question_lst }


# 4지선다형 문제들 로직 체크
def MultipleChoiceQuestion_logic_check(model , question):

    choices = ["choice_1" , "choice_2" , "choice_3" , "choice_4"]

    for choice in choices:
        model.invoke({'question' : question['question'] , 'choice' : choice})
    

# 검증 노드
def logic_verify(state):

    Flag = True

    return {'node_name' : NODE_LOGIC_VERIFY , 'graph_flow' : (state.get('graph_flow') + [NODE_LOGIC_VERIFY]) , 'logic_test' : Flag}

# 로직 테스트 - 라우팅함수
def logic_routing(state):
    if state.get('logic_test' , True):
        return NODE_PROPER_VERIFY
    else:
        return NODE_RAG

# 검증 노드
def proper_verify(state):

    Flag = True
    return {'node_name' : NODE_PROPER_VERIFY , 'graph_flow' : (state.get('graph_flow') + [NODE_PROPER_VERIFY]) , 'proper_test' : Flag}


# 적합 테스트 - 라우팅함수
def proper_routing(state):
    if state.get('proper_test' , True):
        return NODE_SAVE_RESULT
    else:
        return NODE_RAG

# 생성한 문제 저장 및 출력
def save_result(state):

    state['graph_flow'] += [NODE_SAVE_RESULT]

    for model_name in state['model_response']:
        
        print(f"✅ 답변 모델: {model_name}")
        print(f"✅ 토큰 발생 비용: {state['model_response'][model_name]['cost']}")

        questions = state['model_response'][model_name]['questions']
        print_questions_pretty(questions)
        
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M")
        file_name = f"{generate_save_path}/{state['exam_name']}_{state['topic']}_{datetime_str}.json"
        
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=4)
       
        print(f"✅ JSON 파일 저장 완료: {file_name}")    
        return {'node_name' : NODE_SAVE_RESULT} 


        
def print_questions_pretty(questions):
    
    print(f"📘 [{questions[0]['exam_name']}] - {len(questions)} 문항")
    print("-" * 60)
    for q in questions:  
        print(f" 주제: {q['topic']} / 난이도: {q['difficulty']}")
        print(f"📝 Q{q['num']}. {q['question']}")
        print(f"   A. {q['choice_1']}")
        print(f"   B. {q['choice_2']}")
        print(f"   C. {q['choice_3']}")
        print(f"   D. {q['choice_4']}\n")
        print(f"   해설 :  {q['answer']}")
        print("-" * 60)



def main_graph():

    # StateGraph 생성
    builder = StateGraph(State)

    ## 노드 설정 ##
    builder.add_node( NODE_RAG, RunnableLambda(retrieve_context))       # RAG 검색
    builder.add_node( NODE_PROMPT , RunnableLambda(prompt_node))        # 프롬프트
    builder.add_node( NODE_GENERATE , parallel_models)                  # LLM 응답생성  
    builder.add_node( NODE_LOGIC_VERIFY , logic_verify)                 # 논리검증
    builder.add_node( NODE_PROPER_VERIFY , proper_verify)               # 적합성 검증 
    builder.add_node( NODE_SAVE_RESULT , save_result)                   # 결과 저장 및 출력

    ## 엣지 설정 ##
    builder.set_entry_point(NODE_RAG)
    builder.add_edge(NODE_RAG, NODE_PROMPT)
    builder.add_edge(NODE_PROMPT, NODE_GENERATE)
    builder.add_edge(NODE_GENERATE, NODE_LOGIC_VERIFY)
    builder.add_edge(NODE_SAVE_RESULT, END)


    ## 조건부 엣지 설정 ##
    builder.add_conditional_edges( NODE_LOGIC_VERIFY, logic_routing ,
        { NODE_PROPER_VERIFY : NODE_PROPER_VERIFY ,   NODE_RAG: NODE_RAG }
    )

    ## 조건부 엣지 설정 ##
    builder.add_conditional_edges( NODE_PROPER_VERIFY, proper_routing ,
        { NODE_SAVE_RESULT : NODE_SAVE_RESULT ,  NODE_RAG : NODE_RAG }
    )

    # 그래프 컴파일
    app = builder.compile(checkpointer = MemorySaver())
    
    # visualize_graph(app)
    
    # LangGraph 시각화 저장
    img = Image(app.get_graph().draw_mermaid_png())

    with open("MainGraph.png", "wb") as f:
        f.write(img.data)

    return app




