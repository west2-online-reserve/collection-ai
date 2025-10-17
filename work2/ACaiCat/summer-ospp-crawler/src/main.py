import json
import os
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pandas

from project import Project

BASEURL: str = "https://summer-ospp.ac.cn/"
APPLICATION_DIR = Path("applications/")

client = httpx.Client(base_url=BASEURL, event_hooks={
    "request": [lambda request: print(f"{request.method} {request.url}")],
    "response": [lambda response: None if response.is_success else print(
        f"[Error]{response.request.method} {response.url} ({response.status_code}): \n{response.text}")]
})


def get_project_details(project: Project):
    payload: dict[str, Any] = {
        "programId": project.program_code,
        "type": "org"
    }
    json_response: dict[str, Any] = client.post("/api/getProDetail", json=payload,
                                                timeout=30.0).json()

    project.program_desc = json_response["programDesc"]
    project.tech_requirement = [i["title"] for i in json_response["techRequirement"] if i]
    project.org_program_id = json_response["orgProgramId"]

    payload: dict[str, Any] = {
        "proId": project.org_program_id
    }

    response: httpx.Response = client.post("/api/publicApplication", json=payload)

    with open(APPLICATION_DIR.joinpath(f"{project.program_code}.pdf"), mode="wb") as file:
        file.write(response.read())
        file.flush()


def main():
    os.makedirs("applications", exist_ok=True)

    payload: dict[str, Any] = {
        "difficulty": [],
        "lang": "zh",
        "orgName": [],
        "pageNum": 1,
        "pageSize": 50,
        "programName": "",
        "programmingLanguageTag": [],
        "supportLanguage": [],
        "techTag": []
    }

    json_response: dict[str, Any] = client.post("/api/getProList", json=payload,
                                                timeout=30.0).json()

    projects: list[Project] = []
    for row in json_response["rows"]:
        project = Project(row["programName"], row["programCode"], row["difficulty"], json.loads(row["techTag"]), "", [],
                          "")
        projects.append(project)

    df = pandas.DataFrame([i.to_dict() for i in projects])

    with ThreadPoolExecutor(max_workers=10) as pool:
        pool.map(get_project_details, projects)

    df.to_csv("projects.csv", index=False)
    print(f"Export {len(df)} records...")


if __name__ == '__main__':
    main()
