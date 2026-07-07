from pydantic import BaseModel
from typing import Dict, List, Optional
from pydantic import Field
import logging

logger = logging.getLogger(__name__)

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
    
    def __init__(self, **data):
        super().__init__(**data)
        logger.info(f"ProjectSummary created: root={self.root}, total_files={self.total_files}, languages={len(self.languages)}, frameworks={self.frameworks}")


class Reference(BaseModel):
    file_path: str = Field(description="File containing the reference.")
    line: int = Field(description="1-based line number.")
    column: int = Field(description="1-based column number.")
    text: str = Field(description="Line text containing the reference.")


class FindReferencesResult(BaseModel):
    references: list[Reference] = Field(default_factory=list)
    count: int = Field(description="Total references found.")

    def __str__(self) -> str:
        """Formats the structured references into a clean readout for the LLM window."""
        if not self.references:
            return "No reference matches identified across the codebase track."

        lines = [f"Found {self.count} reference(s):"]
        for idx, ref in enumerate(self.references, start=1):
            location_tag = f"{ref.file_path}:{ref.line}:{ref.column}"
            lines.append(f"  {idx}. {location_tag} -> {ref.text.strip()}")
        
        return "\n".join(lines)