from enum import Enum


class ProblemType(str, Enum):
    arithmetic = "arithmetic"
    algebra = "algebra"
    geometry = "geometry"
    coordinate_geometry = "coordinate_geometry"
    trigonometry = "trigonometry"
    calculus = "calculus"
    statistics = "statistics"
    probability = "probability"
    functions = "functions"
    general = "general"


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"
