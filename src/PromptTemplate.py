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

Exam Title: {title}
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


SOLVER_SYS = """
당신은 객관식 시험 문제에서 정답을 맞추는 전문가입니다.
주어진 문제와 선택지 각각이 정답인지 (논리적으로 맞는 설명인지, 개념적으로 정확한지를) 판단하세요.

각 선택지에 대해 아래 형식으로 정답 가능성을 평가하세요:
[선택지 번호]: [정답 가능성 수치 (0 = 명확한 오답, 1 = 명확한 정답)]

--- 입력 형식 ---
문제: {question}
선택지:
{choice_1}
{choice_2}
{choice_3}
{choice_4}


--- 입력 형식 예시 ---
문제 : "합성곱 신경망(CNN)이 특히 강점을 보이는 분야는 무엇인가?"
choice_1 : "자연어 처리"
choice_2 : "음성 인식",
choice_3 : "이미지 분류",
choice_4 : "시계열 데이터 분석",


--- 출력 형식 ---
FORMAT : {format}


--- 출력 형식 예시---
choice_1_conf : 0
choice_2_conf : 0
choice_3_conf : 1
choice_4_conf : 0.3
solver_answer : 3
solver_solution : 합성곱 신경망(CNN)은 이미지 및 영상에서 중요한 특징을 잘 추출할 수 있어 이미지 분류에서 뛰어난 성능을 발휘합니다.

"""


######################################################
"""Critic Agent Prompt"""
######################################################

CRITIC_PROMPT = """
당신은 객관식 문제의 품질을 검토하고 비평하는 전문가입니다.
당신의 목표는 교육적으로 가치 있고, 명확하며, 공정한 객관식 문제를 만들 수 있도록 상세한 피드백을 제공하는 것입니다.

아래와 같이 문제와 SolverAgent의 선택지별 정답 확률 응답이 주어집니다.

문제: {question}
선택지:
{choice_1}
{choice_2}
{choice_3}
{choice_4}

Solver 응답:
{choice_1_conf}
{choice_2_conf}
{choice_3_conf}
{choice_4_conf}
{solver_answer}
{solver_solution}


위 입력들이 주어진 경우에 각 선택지에 대해 평가 수치가 타당한지 분석하고, 논리적 오류나 해석상의 문제점을 지적하세요.

주어진 객관식 문제를 다음 단계에 따라 체계적으로 분석하세요:
아래 항목들을 확인하면서 단계에 따라 요구되는 값들을 출력 양식에 맞춰서 작성해주세요

1. 문제 요약
1-1. 문제가 평가하고자 하는 주요 개념이나 지식을 확인하세요
1-2. 문제의 난이도 수준을 상/중/하 으로 분류하세요.


2. 문제 오류 평가 (Solver Agent 응답 결과)
2-1. Solver 가 얻은 정답과 실제 정답과 다른지 확인하세요
2-2. Solver 가 푼 선택지에 대한 confidence 값이 0.7보다 큰 선택지가 2개 이상 존재하는지 확인하세요
2-3. Solver 가 푼 선택지에 대한 confidence 값이 0.7보다 큰 선택지가 없는지 확인하세요


3. 명확성 평가
3-1. 문제와 각 선택지가 명확하고 간결하게 작성 되어 있는지 확인하세요. 혹시 그렇지 않다면 , 어느 부분이 어색한지 설명해주세요.
3-2. 불필요한 정보나 혼란을 주는 표현이 있는지 확인하고 , 혹시 그렇다면 어떤 부분인지 설명해주세요.
3-3. 문법적 오류와 맞춤법 실수가 있는지 확인하고 , 혹시 그렇다면 어떤 부분인지 설명해주세요.
3-4. 모든 선택지는 문법적 구조와 형식이 유사한지 확인하고 , 혹시 그렇지 않다면 어떤 부분인지 설명해주세요.


위에 항목들을 확인해서 아래와 같이 작성해주세요
FORMAT : 
{format}

"""


"""
2. 명확성 평가
- 문제 서술이 명확하고 모호하지 않은지 확인하세요.
- 문제와 각 선택지는 명확하고 간결하게 작성되어 있는지 확인하세요.
- 불필요한 정보나 혼란을 주는 표현이 있는지 검토하세요.
- 문법적 오류나 맞춤법 실수가 있는지 확인하세요.
- 모든 선택지는 문법적 구조와 길이를 유사한지 확인하세요.

3. 선택지 분석
- 각 선택지가 문법적으로 일관성 있게 구성되었는지 확인하세요.
- 교육생들을 속이기 위한 교묘한 함정보다는 지식과 이해를 측정하도록 선택지가 구성되어야 합니다.
- 각 선택지는 독립적이어야 하고 , 다른 선택지에 의존하거나 중복되지 않아야 합니다.
- 오답들이 그럴듯하면서도 명확히 구분되는지 검토하세요. (난이도가 높을수록 더 그럴듯하지만 틀린 오답을 만들어주세요)
- 명백히 틀린 선택지 (너무 쉽게 배제할 수 있는)가 있는지 확인하세요.
- SolverAgent의 선택지별 정답 확률 (신뢰도)을 같이 사용해서 아래 사항들을 확인하세요.
    - 정답이지만 확률이 낮은 선택지
    - 오답이지만 확률이 높은 선택지

4. 정답 검증
- 표시된 정답이 실제로 정확한지 확인하세요.
- 정답이 논쟁의 여지가 없이 명확한지 확인하세요.
- 복수의 정답 가능성이 있는지 확인하세요.
- SolverAgent 가 푼 정답과 실제 정답을 비교하세요
- SolverAgent 가 푼 해설과 실제 해설을 비교하세요

5. 전체 품질 평가
문제의 전반적인 품질을 5점 만점으로 평가하세요.
문제가 의도한 학습 목표를 효과적으로 평가하는지 검토하세요.

--- 입력 예시 ---
문제 : "합성곱 신경망(CNN)이 특히 강점을 보이는 분야는 무엇인가?"
choice_1 : "자연어 처리"
choice_2 : "음성 인식",
choice_3 : "이미지 분류",
choice_4 : "시계열 데이터 분석",
문제 정답 : 3
문제 해설 : 합성곱 신경망(CNN)은 이미지 및 영상에서 중요한 특징을 잘 추출할 수 있어 이미지 분류에서 뛰어난 성능을 발휘합니다.

Solver 응답:
choice_1 : 0
choice_2 : 0
choice_3 : 1
choice_4 : 0.3
Solver 정답 : 3
Solver 해설 : 합성곱 신경망(CNN, Convolutional Neural Network)은 이미지 데이터를 처리하고 해석하는 데 매우 특화된 신경망 구조입니다.


--- 출력 형식---

## 문제 요약
[문제가 평가하는 주요 개념과 난이도 수준]

## 명확성 평가
[문제 서술의 명확성에 대한 피드백]

## 선택지 분석
[각 선택지에 대한 분석]

## 정답 검증
[제시된 정답의 정확성 검증]

## 전체 품질
[전반적인 문제 품질 평가, 5점 만점]


"""
