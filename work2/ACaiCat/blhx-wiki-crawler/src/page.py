from pydantic import BaseModel


class Page(BaseModel):
    name: str
    page_id: int
    source: str
