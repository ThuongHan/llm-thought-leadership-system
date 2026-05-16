from pydantic import BaseModel
from typing import List


class DimensionScore(BaseModel):
    score: int  
    comment: str

class PostEvaluation(BaseModel):
    post_index: int
    angle: str

    tone_of_voice: DimensionScore
    language_and_style: DimensionScore
    format_adherence: DimensionScore
    style_similarity: DimensionScore

    overall_score: int

class EvaluationResult(BaseModel):
    evaluations: List[PostEvaluation]

    