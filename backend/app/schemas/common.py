from enum import Enum


class ProblemType(str, Enum):
    arithmetic = "arithmetic"
    algebra = "algebra"
    geometry = "geometry"
    trigonometry = "trigonometry"
    calculus = "calculus"
    probability = "probability"
    statistics = "statistics"
    general = "general"


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"
