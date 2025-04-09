from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal
from typing import Annotated, List
from datetime import datetime
from langchain_core.output_parsers import PydanticOutputParser
from typing import Literal, get_args
from typing import Annotated, TypedDict, List, Literal
from langgraph.graph.message import add_messages


# 4지 선다 문제
class MultiChoiceQuestion(BaseModel):

    exam_name : str = Field(description="시험 이름")
    topic : str = Field(description="범위 및 주제")
    num : int = Field(description="문제 번호")
    question : str = Field(description="문항")
    choice_1 : str = Field(description="첫번째 선택지")
    choice_2 : str = Field(description="두번째 선택지")
    choice_3 : str = Field(description="세번째 선택지")
    choice_4 : str = Field(description="네번째 선택지")
    answer : int = Field(description="몇 번 선택지가 정답인지")
    solution : int = Field(description="문제에 대한 해설을 명확하고 간결하게 설명해주세요")
    
    difficulty : Literal['하', '중', '상']  = Field(description=
                             """
                            "문제 난이도\n"
                            "- 하: 기본적인 개념 이해\n"
                            "- 중: 기본 개념을 이해하고 상황에 맞게 적용 가능"
                            "- 상: 복합적인 사고/추론/적용 필요""")
                    


class ragState(TypedDict):
    
    file_path : str  # RAG 문서 경로
    query : str # 검색 쿼리
    
    ## RAG 관련 ##
    input_docu : str
    use_rag: bool  # RAG 사용 여부
    rag_option : int 
    context: str   # RAG에서 검색된 문맥 정보

    ## 메세지 ##
    messages: Annotated[list, add_messages]


    
## ✅ 1. 상태 정의 ##
class State(TypedDict):

    ## 노드 정보 ##
    node_name : str
    
    ## 모델 정보 ##
    models : dict
    models_info : dict
    
    ## 문제 생성에 필요한 정보들 ##
    exam_name : str
    topic : str
    difficulty: str
    num_question: int

    input_file_path : str
    rag_option : int


    ## RAG 관련 ##
    use_rag: bool  # RAG 사용 여부
    context: str   # RAG에서 검색된 문맥 정보

    ## 메세지 ##
    messages: Annotated[list, add_messages]
    
    ## 프롬프트 메세지 ##
    prompt_message : str
    
    ## LLM 모델 ##
    messages: List
    model_response: dict  # 각 모델의 응답을 저장할 딕셔너리

    ## 생성한 문제정보 ##
    question : List[MultiChoiceQuestion]

    ## 검증 정보 ##
    logic_test : bool
    proper_test : bool
    
    ## 그래프 동작 확인 ##
    graph_flow : List
    node_info : dict
    
class LogicCheck(BaseModel):
    is_error : bool = Field(description = "논리적으로 맞는 말인지?(True/False)")
    is_confused : bool = Field(description = "혼동되거나 애매하거나 잘못 이해해서 풀 여지가 있는지?")


if __name__ == "__main__":
    
    # 예시 데이터
    question_data = {
        "exam_name": "ADsP",
        "topic": "데이터 분석",
        "num": 1,
        "question": "데이터 분석의 기본 개념은 무엇인가요?",
        "choice_1": "데이터 수집",
        "choice_2": "데이터 전처리",
        "choice_3": "데이터 분석",
        "choice_4": "모델링",
        "answer": "데이터 분석은 데이터를 수집하고, 전처리하고, 분석하여 인사이트를 도출하는 과정입니다.",

    }
    
    # MultiChoiceQuestion 모델 생성
    question = MultiChoiceQuestion(**question_data)
    print(question.model_dump_json(indent=4))  # JSON 형식으로 출력