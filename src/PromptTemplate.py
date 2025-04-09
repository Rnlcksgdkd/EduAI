from langchain_core.prompts import ChatPromptTemplate


# 기존 프롬프트 템플릿
systemTemplate = """
You're an exam question creator.
Your goal is to design questions that help students fully understand key concepts so they can perform well on their exams.
Your questions should be concept-driven and designed to reinforce understanding.
Please respond in Korean.
"""

questionTemplate = """
Please Create {num_question} questions with four options (A–D), the correct answer, and an explanation based on the conditions below:
"If a CONTEXT value exists, please generate questions based on its content."

The output must be a single List(NOT JSON) containing JSON objects. Each object must conform to the JSON schema provided below. 
Do not output separate JSON objects or any extra text outside of this single list
AND do not use Code Block!

Exam Name: {exam_name}
Topic : {topic}
CONTEXT : {context}
FORMAT :
{format}


"""

standard_prompt_template = ChatPromptTemplate.from_messages([
    ("system", systemTemplate),
    ("human", questionTemplate)
])


Logic_Check_systemTemplate = """
당신은 객관식 시험 문제의 선택지를 검토하는 전문가입니다. 
당신의 역할은 다음과 같습니다:
0. 문제과 그에 대한 선택지 쌍이 당신에게 주어집니다.
1. 주어진 문제와 선택지 쌍이 논리적으로 타당하고 개념적으로 정확한지를 판단합니다.
2. 해당 선택지가 어느정도 정답이 될 수 있는지를 0부터 1까지의 값으로 수치화합니다. (0 = 명확한 오답 , 1 = 명확한 정답)
3. 판단한 이유를 간결하고 논리적으로 설명합니다.

답변은 간결하고 논리적으로 작성하세요.
아래는 질문/선택지 에 대한 답변 예시입니다.

====================================================================================================
문제). 정규분포의 특징으로 옳지 않은 것은 무엇인가?
선택지). 정규분포는 대칭적이다.

정답 여부 : False 
정답 가능성 : 0.0
로직 체크 결과 : 정규분포는 대칭적인 특징을 가지므로 이 선택지는 정답이 아니다.
====================================================================================================
문제). 정규분포의 특징으로 옳지 않은 것은 무엇인가?
선택지). 정규분포의 평균, 중앙값, 최빈값은 같다.

정답 여부 : False 
정답 가능성 : 0.0
로직 체크 결과 : 정규분포의 평균, 중앙값, 최빈값은 모두 같다는 것은 정규분포의 핵심 특성이므로 이 선택지는 옳고 옳지 않은 것을 고르는 문제이므로 정답이 아닙니다.
====================================================================================================
====================================================================================================
문제). 정규분포의 특징으로 옳지 않은 것은 무엇인가?
선택지). 정규분포의 분포는 비대칭적이다.

정답 여부 : True 
정답 가능성 : 1.0
로직 체크 결과 : 정규분포는 대칭적인 분포로, 평균을 기준으로 좌우가 동일한 모양을 가지므로 '비대칭적이다'는 선택지는 옳지 않은것이므로 정답입니다.
====================================================================================================
"""

Logic_Check_questionTemplate =  """
     다음은 하나의 질문과 그에 대한 선택지로 구성된 쌍입니다.\n\n'
     '"질문": "{question}"\n'
     '"선택지": "{answer}"\n\n'
    
     '아래의 형식에 맞춰 반드시 답변해주세요:\n\n'
      AND do not use Code Block! AND return dictionary
      FORMAT : {format}
"""

Logic_Check_promptTemplate = ChatPromptTemplate.from_messages([
    ("system", Logic_Check_systemTemplate),
    ("human", Logic_Check_questionTemplate)
])
