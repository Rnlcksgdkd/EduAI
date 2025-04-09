import os
import sys
from langchain_core.runnables import RunnableLambda, RunnableMap, RunnableConfig
from dotenv import load_dotenv

### src 모듈 import ###
sys.path.append(os.path.dirname(os.path.abspath(__file__).replace('\\' , '/')) + '/src')
import MainGraph
from RagManager import RAGManager 
from ModelManager import ModelManager
import PromptTemplate
import Structure


if __name__ == "__main__":
    
    
    ## .env 파일 로드
    load_dotenv('./.env')
        
    # 모델 선택
    model_manager = ModelManager(['gpt-4o-mini' , 'gpt-4o'])

    # 입력 생성
    input_dict = {
    "models" : model_manager.models,
    "models_info" : model_manager.models_info,
    "exam_name": "빅데이터분석기사",
    "topic" : "머신러닝" ,
    "num_question": 10,
    "use_rag": False
    }

    # Config 설정
    config = RunnableConfig(
        recursion_limit=10,  # 최대 10개의 노드까지 방문. 그 이상은 RecursionError 발생
        configurable={"thread_id": "7"},  # 스레드 ID 설정
        tags=["my-tag"],  # Tag
    )

    # Graph 생성
    app = MainGraph.main_graph()
    
    # Event 저장 리스트
    test_list = []
    event_list = []
    
    # Graph 실행
    for event in app.stream(input=input_dict, config=config):
        event_list.append(event)