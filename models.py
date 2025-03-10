from pydantic import BaseModel


class ConvFinQAEntry(BaseModel):
    pre_text: list
    post_text: list
    table: list


class AgentResponse(BaseModel):
    """Structured response for financial questions"""

    answer: str
    calculation_explanation: str
    data_points_used: list[str]


class ResponseMetadata(AgentResponse):
    expected_answer: str
    percentage_error: float
    question: str
    model_choice: str
