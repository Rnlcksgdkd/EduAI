from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from typing import Dict

class MyState(BaseModel):
    context: dict = Field(default_factory=lambda: {"topic": "과학", "level": "중급"})

def add_topic_info(state: MyState) -> MyState:
    print("현재 토픽:", state.context["topic"])
    return state

builder = StateGraph(MyState)
builder.add_node("show_topic", add_topic_info)
builder.set_entry_point("show_topic")
graph = builder.compile()

result = graph.invoke(MyState())
