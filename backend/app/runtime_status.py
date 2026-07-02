from pydantic import BaseModel


class RuntimeDependencyStatus(BaseModel):
    name: str
    available: bool
    version: str | None = None
    detail: str | None = None
    install_hint: str | None = None
    required_for: list[str] = []


class RuntimeStatus(BaseModel):
    ready: bool
    dependencies: list[RuntimeDependencyStatus]
