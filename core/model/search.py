from pydantic import BaseModel
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class ProjectSummary(BaseModel):
    root: str = Field(description="The absolute resolved path of the repository root.")
    total_files: int = Field(description="Total count of non-ignored files scanned.")
    languages: Dict[str, int] = Field(description="Frequency map of detected languages.")
    frameworks: List[str] = Field(description="Detected frameworks or language environments.")
    package_manager: Optional[str] = Field(description="Primary project package manager detected.")
    top_level_directories: List[str] = Field(description="List of core root level directories.")
    source_directories: List[str] = Field(description="List of detected codebase execution source tracks.")
    test_directories: List[str] = Field(description="List of identified testing suites locations.")
    config_files: List[str] = Field(description="Configuration files detected across the tree layout.")