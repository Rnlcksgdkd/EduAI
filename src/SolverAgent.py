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

## Solver Agent 클래스 정의
class SolverAgent():
    
    def __init__(self, model_name = 'gpt-4o'):
        
        # Initialize the LLM model
        self.model = ChatOpenAI(model=model_name)
        self.parser = PydanticOutputParser(pydantic_object=Structure.OutputSolver)
        
    # batch 처리 함수
    def run_batch(self):
        return RunnableMap({
            "question_info" : lambda s : s['question_info'],
            "solver_answer": RunnableLambda(self.generate_response)
        })

    def complete_prompt(self , question):
        
        template = ChatPromptTemplate.from_messages([('system' , PromptTemplate.SOLVER_SYS)])
        prompt = template.format_messages( **{**question , 'format' : self.parser.get_format_instructions() })
        
        return prompt


    def generate_response(self , state):
        
        question_info = state['question_info']
        input_message = self.complete_prompt(question_info)
        response = self.model.invoke(input_message).content
        output = self.parser.parse(response)
        return output
    
    
if __name__ == "__main__":
    

    
    # Define a sample question
    question_1 = {
      "question": "데이터 전처리에서 결측치를 처리하는 방법은 무엇인가?",
    "choice_1": "제거",
    "choice_2": "대체",
    "choice_3": "무시",
    "choice_4": "정규화",
    }
    
    
    
    # 질문 배치 생성 (예시)
    question_batch = [ {"question_info" : question_1}]
    
    
    # Example usage
    solver_agent = SolverAgent()
    runner = solver_agent.run_batch()
    results = [runner.invoke(q) for q in question_batch]
    
    
    for r in results:
        print(f"Q: {r['question_info']}")
        print(f"A: {r['solver_answer']}\n")