from pydantic import BaseModel, Field


class SimpleResponse(BaseModel):
    answer: str = Field(description="The answer to the riddle")


class ReasoningResponse(BaseModel):
    reasoning: str = Field(description="The step-by-step reasoning process to arrive at the answer")
    answer: str = Field(description="The answer to the riddle")
