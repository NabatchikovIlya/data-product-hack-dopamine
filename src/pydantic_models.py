from enum import Enum

from pydantic import BaseModel


class Level(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class PsychoLevel(Enum):
    NORMAL = "NORMAL"
    ABNORMAL = "ABNORMAL"


class BaseScore(BaseModel):
    value: float


class Score(BaseScore):
    level: Level


class PsychoScore(BaseScore):
    level: PsychoLevel


class TestScores(BaseModel):
    decentration: Score
    empathy: Score
    empathic_care: Score
    empathic_distress: Score
    psychological_indicator: PsychoScore
    total: Score


class ServiceOutput(BaseModel):
    test_scores: TestScores
    dopamine: float
