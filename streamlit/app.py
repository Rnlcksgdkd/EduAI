import streamlit as st
import json

# JSON 파일 경로 또는 dict 직접 사용
with open("../feedback/AI__20250415_1740.json", "r", encoding="utf-8") as f:
    data = json.load(f)

questions = data["Question"]
solvers = data["Solver"]
critics = data["Critic"]

st.set_page_config(layout="wide")
st.title("문제 검토 시각화 시스템")

for idx in range(len(questions)):
    q = questions[idx]
    s = solvers[idx]
    c = critics[idx]

    st.markdown("---")
    cols = st.columns([2, 1])

    # 왼쪽 영역 - 문제 정보
    with cols[0]:
        st.markdown(f"### {q['num']}번 문제 ｜ 객관식 ｜ 난이도 - {q['difficulty']}")
        st.markdown(f"**{q['question']}**")

        for i in range(1, 5):
            choice = q[f"choice_{i}"]
            st.markdown(f"**{i}번.** {choice}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Solver 정답 및 확신도 요약
        score_text = " ".join([
            f"{i}번 ({s.get(f'choice_{i}_conf', '0')})" for i in range(1, 5)
        ])
        st.markdown(f"#### 🧠 **Solver의 풀이** {score_text}")
        st.markdown(s["solver_solution"])

    # 오른쪽 영역 - Critic
    with cols[1]:
        st.markdown("### 🧐 Critic의 피드백")
        st.markdown(f"- **개념:** {c['question_concept']}")
        st.markdown(f"- **난이도 판단:** {c['question_level']}")
        st.markdown(f"- **오답 포함 여부:** {'없음' if not c['wrong_answer'] else '있음'}")
        st.markdown(f"- **복수 정답 가능성:** {'없음' if not c['multiple_plausible'] else '있음'}")
        st.markdown(f"- **명확한 정답 없음:** {'없음' if not c['no_clear_answer'] else '있음'}")
        st.markdown(f"- **표현 명확성:** {c['question_clear']}")
        st.markdown(f"- **불필요한 표현:** {c['question_unnecessary']}")
        st.markdown(f"- **문법 및 맞춤법:** {c['question_grammar']}")
        st.markdown(f"- **선택지 구조:** {c['question_structure']}")
