from enum import Enum


class ProblemType(str, Enum):
    arithmetic = "arithmetic"
    algebra = "algebra"
    number_theory = "number_theory"
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
