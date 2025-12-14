from dataclasses import dataclass, field
from typing import cast
import json

@dataclass
class ProjectOutputRequirement:
    title: str
    children: list[object] = field(default_factory=list)

@dataclass
class Project:
    # From list
    program_id: str  # programCode
    program_name: str # programName
    difficulty: str # difficulty
    tech_tags: list[str] # techTag parsed
    org_id: str # orgId
    
    # From detail
    program_desc: str | None = None # programDesc
    output_requirements: list[ProjectOutputRequirement] = field(default_factory=list) # outputRequirement parsed
    org_program_id: int | None = None # orgProgramId (needed for PDF)

    def to_dict(self):
        return {
            "program_id": self.program_id,
            "program_name": self.program_name,
            "difficulty": self.difficulty,
            "tech_tags": self.tech_tags,
            "org_id": self.org_id,
            "program_desc": self.program_desc,
            "output_requirements": [req.__dict__ for req in self.output_requirements],
            "org_program_id": self.org_program_id
        }

def parse_tech_tags(tag_str: str) -> list[str]:
    """Parses the nested JSON string for tech tags."""
    if not tag_str:
        return []
    try:
        # tag_str looks like '[["os","Linux"],["codelang","Programming Language"]]'
        tags = cast(list[object], json.loads(tag_str))
        # Extract the last element of each inner list which seems to be the specific tag name
        # e.g. "Linux", "Programming Language"
        flattened: list[str] = []
        for t in tags:
            if isinstance(t, list):
                # Expecting t to be [category, tag_name]
                t_list = cast(list[object], t)
                if len(t_list) > 1:
                    val = t_list[1]
                    if isinstance(val, str):
                        flattened.append(val)
            elif isinstance(t, str):
                flattened.append(t)
        return flattened
    except json.JSONDecodeError:
        return []
