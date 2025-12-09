from pydantic import BaseModel, Field


class SimpleResponse(BaseModel):
    answer: str = Field(
        description="The answer to the riddle. Provide only the single word or short phrase that answers the riddle, without any additional text or punctuation."  # noqa: E501
    )


class ReasoningResponse(BaseModel):
    reasoning: str = Field(description="The step-by-step reasoning process to arrive at the answer")
    answer: str = Field(
        description="The answer to the riddle. Provide only the single word or short phrase that answers the riddle, without any additional text or punctuation."  # noqa: E501
    )
