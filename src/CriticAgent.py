from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

## .env 파일 로드
load_dotenv('../.env')

from langchain_openai import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableMap
import Structure
import PromptTemplate

ErrorCASE = {
    
    'wrong_answer' : """정답이 다른 선택지인 경우
                        Solver가 얻은 정답이 실제정답과 다름""",
    
    'multiple_plausible' :      """ 복수 정답 가능성인 경우
                                Solver의 선택지에 대한 confidence 값이 0.7보다 큰 선택지가 2개 이상 존재""" , 
    
    'no_clear_answer' : """ 정답이 없는 경우
                            Solver가 푼 선택지에 대한 confidence 값이 0.7 보다 큰 선택지가 없는 경우 """ ,
    
}


### Solver Agent Output Format ###
class OutputCritic(BaseModel):
    
    
    # 1. 문제 요약
    question_concept : str = Field(description="문제가 평가하고자 하는 주요 개념이나 지식")
    question_level : str = Field(description="문제의 난이도 수준")
    
    
    # 2. 문제 오류 평가
    wrong_answer         : bool = Field(description= ErrorCASE['wrong_answer'] )
    multiple_plausible   : bool = Field(description= ErrorCASE['multiple_plausible'] )
    no_clear_answer      : bool = Field(description= ErrorCASE['no_clear_answer'] )


    # 3. 명확성 평가
    question_clear : str = Field(description="문제와 각 선택지가 명확하고 간결하게 작성 되어 있는지 확인하고, 혹시 그렇지 않다면 어느 부분인지 설명해주세요.")
    question_unnecessary : str = Field(description="불필요한 정보나 혼란을 주는 표현이 있는지 확인하고 , 혹시 그렇다면 어떤 부분인지 설명해주세요.")
    question_grammar : str = Field(description="문법적 오류와 맞춤법 실수가 있는지 확인하고 , 혹시 그렇다면 어떤 부분인지 설명해주세요.")
    question_structure : str = Field(description="모든 선택지는 문법적 구조와 형식이 유사한지 확인하고 , 혹시 그렇지 않다면 어떤 부분인지 설명해주세요.")
    
    
    
    
## Solver Agent 클래스 정의
class CriticAgent():
    
    def __init__(self, model_name = 'gpt-4o'):
        
        # Initialize the LLM model
        self.model = ChatOpenAI(model=model_name)
        self.parser = PydanticOutputParser(pydantic_object=OutputCritic)
        
        
    def complete_prompt(self , question , solver_answer):
        
        template = ChatPromptTemplate.from_messages([('system' , PromptTemplate.CRITIC_PROMPT)])
        prompt = template.format_messages( **{**question , **solver_answer , 'format' : self.parser.get_format_instructions() })
        
        return prompt

    def generate_response(self , state):
        
        question_info = state.get('question_info' , "")
        solver_answer = state.get('solver_answer' , "")
        
        input_message = self.complete_prompt(question_info , solver_answer)
        response = self.model.invoke(input_message).content
        output = self.parser.parse(response)
        return output
    
    
        # batch 처리 함수
    def run_batch(self):
        return RunnableMap({
            "question_info" : lambda s : s['question_info'],
            "critic_answer": RunnableLambda(self.generate_response)
        })



if __name__ == "__main__":
    
    test_path = '../feedback/AI__20250410_0956.json'
    
    import json
    with open(test_path ,'r' , encoding='utf-8') as f:
        batch_data = json.load(f)

    question_batch = batch_data['Question']
    solver_batch = batch_data['Solver']

    # Example usage
    critic_agent = CriticAgent()
    runner = critic_agent.run_batch()
    
    for i in range(len(question_batch)):
        question = question_batch[i]
        solver_answer = solver_batch[i]
        result = runner.invoke({'question_info' : question  , 'solver_answer' : solver_answer})
        print(result)
    
    
