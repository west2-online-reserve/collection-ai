from dataclasses import dataclass


@dataclass
class Project:
    name: str
    program_code: str
    difficulty: str
    tech_tag: list[str]
    program_desc: str
    tech_requirement: list[str]
    org_program_id: str

    def to_dict(self):
        return {
            "name": self.name,
            "program_code": self.program_code,
            "difficulty": self.difficulty,
            "tech_tag": self.tech_tag,
            "program_desc": self.program_desc,
            "tech_requirement": self.tech_requirement,
            "org_program_id": self.org_program_id
        }
