from pydantic import BaseModel

class ProjectSummary(BaseModel):
    root: str
    total_files: int
    languages: dict[str, int]
    frameworks: list[str]
    package_manager: str | None
    top_level_directories: list[str]
    source_directories: list[str]
    test_directories: list[str]
    config_files: list[str]