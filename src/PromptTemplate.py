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

The output must be a single List(NOT JSON) containing JSON objects. Each object must conform to the JSON schema provided below. 
Do not output separate JSON objects or any extra text outside of this single list
AND do not use Code Block!

Exam Name: {exam_name}
Topic : {topic}

FORMAT :
{format}
"""

standard_prompt_template = ChatPromptTemplate.from_messages([
    ("system", systemTemplate),
    ("human", questionTemplate)
])

Logic_Check_systemTemplate = """
    너는 질문과 대답의 논리적 타당성을 평가하는 판단자 역할을 맡고 있어.
    사용자가 제시하는 '질문'과 '대답' 쌍이 논리적으로 일관된지, 혼동의 여지는 없는지를 판단해줘.
"""

Logic_Check_questionTemplate =  """
     다음은 하나의 질문과 그에 대한 대답으로 구성된 쌍입니다.\n\n'
     '"질문": "{question}"\n'
     '"대답": "{answer}"\n\n'
     '이 쌍이 논리적으로 정확한지 판단해주세요.\n\n'
     '질문의 맥락과 대답이 정확하고 자연스럽고 일관성 있는지 평가해 주세요.\n\n'
     
     '아래의 형식에 맞춰 반드시 답변해주세요:\n\n'
      FORMAT : {format}
"""

Logic_Check_promptTemplate = ChatPromptTemplate.from_messages([
    ("system", Logic_Check_systemTemplate),
    ("human", Logic_Check_questionTemplate)
])
