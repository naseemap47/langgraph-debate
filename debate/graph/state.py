from typing import TypedDict, List, Dict, Optional, Any

class DebateState(TypedDict):
    topic: str
    active_models: List[str]
    judge_model: str
    current_stage: str
    stage_1_answers: Dict[str, str]
    stage_2_critiques: Dict[str, str]
    stage_3_synthesis: Optional[str]
    status_events: List[Dict[str, Any]]
    errors: Dict[str, str]
