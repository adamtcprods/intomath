from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.geometry_dsl import GeometryDSL

_SUPPORTED_IMAGE_MIME_TYPES = frozenset(
    ["image/jpeg", "image/png", "image/webp", "image/gif"]
)


def strip_latex_wrappers(value: str) -> str:
    expression = value.strip()
    if expression.startswith("```") and expression.endswith("```"):
        expression = expression.removeprefix("```").removesuffix("```").strip()
        for language in ("latex", "tex", "math"):
            if expression.lower().startswith(language):
                expression = expression[len(language) :].strip()
                break

    delimiter_pairs = (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$"))
    changed = True
    while changed:
        changed = False
        for opener, closer in delimiter_pairs:
            if expression.startswith(opener) and expression.endswith(closer):
                expression = expression[len(opener) : -len(closer)].strip()
                changed = True
    return expression


class ProblemInput(BaseModel):
    text: str = ""
    image_base64: str | None = None
    image_mime_type: str | None = None
    language: str = "auto"

    @model_validator(mode="after")
    def validate_input(self) -> "ProblemInput":
        has_text = bool(self.text.strip())
        has_image = bool(self.image_base64)
        if not has_text and not has_image:
            raise ValueError(
                "At least one of 'text' or 'image_base64' must be provided."
            )
        if has_image and self.image_mime_type is not None:
            if self.image_mime_type not in _SUPPORTED_IMAGE_MIME_TYPES:
                raise ValueError(
                    f"Unsupported image_mime_type '{self.image_mime_type}'. "
                    f"Must be one of: {sorted(_SUPPORTED_IMAGE_MIME_TYPES)}."
                )
        return self


class SolveOptions(BaseModel):
    include_visualization: bool = True


class SolveRequest(BaseModel):
    input: ProblemInput
    options: SolveOptions = Field(default_factory=SolveOptions)


class SolveAnswer(BaseModel):
    text: str
    latex: str | None = None

    @field_validator("text", mode="before")
    @classmethod
    def coerce_text(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return "\n".join(str(item) for item in value if item is not None)
        return str(value)

    @field_validator("latex", mode="before")
    @classmethod
    def coerce_optional_latex(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, list):
            joined = "\n".join(
                strip_latex_wrappers(str(item)) for item in value if item is not None
            )
            return joined or None
        return strip_latex_wrappers(str(value))


class SolveStep(BaseModel):
    index: int
    title: str
    explanation: str
    why_it_happens: str | None = None
    common_mistakes: list[str] = Field(default_factory=list)
    alternative_approaches: list[str] = Field(default_factory=list)
    hints: list[str] = Field(default_factory=list)
    exam_tip: str | None = None
    latex: list[str] = Field(default_factory=list)

    @field_validator("title", "explanation", mode="before")
    @classmethod
    def coerce_required_string_fields(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return "\n".join(str(item) for item in value if item is not None)
        return str(value)

    @field_validator("why_it_happens", "exam_tip", mode="before")
    @classmethod
    def coerce_optional_string_fields(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, list):
            joined = "\n".join(str(item) for item in value if item is not None)
            return joined or None
        return str(value)

    @field_validator(
        "common_mistakes",
        "alternative_approaches",
        "hints",
        mode="before",
    )
    @classmethod
    def coerce_string_list_fields(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        return [str(value)]

    @field_validator("latex", mode="before")
    @classmethod
    def coerce_latex_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [strip_latex_wrappers(value)]
        if isinstance(value, list):
            return [
                strip_latex_wrappers(str(item)) for item in value if item is not None
            ]
        return [strip_latex_wrappers(str(value))]


class SolvePart(BaseModel):
    label: str
    question: str = ""
    answer: SolveAnswer
    steps: list[SolveStep] = Field(default_factory=list)

    @field_validator("label", "question", mode="before")
    @classmethod
    def coerce_string_fields(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return "\n".join(str(item) for item in value if item is not None)
        return str(value)


class GeoGebraPayload(BaseModel):
    commands: list[str] = Field(default_factory=list)
    command_string: str = ""
    validation_passed: bool = True
    issues: list[str] = Field(default_factory=list)


class VisualizationPayload(BaseModel):
    kind: str = "none"
    summary: str | None = None
    dsl: GeometryDSL | None = None
    geogebra: GeoGebraPayload | None = None


class RoutingPayload(BaseModel):
    parser_model: str
    solver_model: str
    vision_model: str | None = None
    reason: str


class SolveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: str = "ok"
    problem_type: str
    difficulty: str
    answer: SolveAnswer
    steps: list[SolveStep]
    parts: list[SolvePart] = Field(default_factory=list)
    visualization: VisualizationPayload
    confidence: float
    routing: RoutingPayload
    cached: bool = False
    warnings: list[str] = Field(default_factory=list)
