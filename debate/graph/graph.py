from typing import Dict
from langgraph.graph import StateGraph, END
from debate.drivers.base import BaseModelDriver
from debate.graph.state import DebateState
from debate.graph.nodes import (
    create_stage_1_node,
    create_stage_2_node,
    create_stage_3_node,
)

def build_debate_graph(drivers: Dict[str, BaseModelDriver]):
    workflow = StateGraph(DebateState)

    workflow.add_node("stage_1_init", create_stage_1_node(drivers))
    workflow.add_node("stage_2_critique", create_stage_2_node(drivers))
    workflow.add_node("stage_3_synthesis", create_stage_3_node(drivers))

    workflow.set_entry_point("stage_1_init")
    workflow.add_edge("stage_1_init", "stage_2_critique")
    workflow.add_edge("stage_2_critique", "stage_3_synthesis")
    workflow.add_edge("stage_3_synthesis", END)

    return workflow.compile()
