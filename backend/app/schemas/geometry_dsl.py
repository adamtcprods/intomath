from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GeometryActionType(str, Enum):
    CREATE_POINT = "CREATE_POINT"
    CREATE_LINE = "CREATE_LINE"
    CREATE_CIRCLE = "CREATE_CIRCLE"
    CREATE_POLYGON = "CREATE_POLYGON"
    INTERSECT = "INTERSECT"
    MIDPOINT = "MIDPOINT"
    PERPENDICULAR = "PERPENDICULAR"
    PARALLEL = "PARALLEL"
    ANGLE_BISECTOR = "ANGLE_BISECTOR"
    CREATE_FUNCTION = "CREATE_FUNCTION"


class GeometryAction(BaseModel):
    model_config = ConfigDict(extra="allow")

    action: GeometryActionType
    label: str | None = None
    points: list[str] = Field(default_factory=list)
    coordinates: tuple[float, float] | None = None
    center: str | None = None
    radius: float | None = None
    through: list[str] | None = None
    line: str | None = None
    equation: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeometryDSL(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "1.0"
    space: str = "euclidean_2d"
    actions: list[GeometryAction] = Field(default_factory=list)
    render_hints: dict[str, Any] = Field(default_factory=dict)
