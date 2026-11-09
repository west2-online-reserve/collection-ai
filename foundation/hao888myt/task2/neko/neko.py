import html
import json
import re

import requests

API = "https://summer.ospp.ac.cn/2025/api"


def get_projects():
    projects = []
    page = 1

    while True:
        print(f"正在获取第{page}页项目列表喵")
        url = API + "/getProList"
        response = requests.post(
            url, json={"pageNum": page, "pageSize": 100}, timeout=20
        )
        data = response.json()

        projects.extend(data["rows"])
        print(f"拿到了{len(projects)}个项目喵")

        if len(projects) >= data["total"]:
            break

        page += 1

    return projects


def clean_html(text: str) -> str:
    # AI写的正则，爽爽筛
    text = re.sub(r"<br\s*/?>", "\n", str(text), flags=re.IGNORECASE)
    text = re.sub(r"</(?:p|li|div)\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    return "\n".join(lines)


def get_tag(detail) -> list:
    tags = detail.get("techTag")
    if not tags:
        return []

    if isinstance(tags, str):
        tags = json.loads(tags)

    result = []
    for tag in tags:
        result.append(tag[1])

    return result


def get_requirement(detail) -> list[str]:
    requirements = detail.get("outputRequirement")
    lines = []

    if not requirements:
        return lines

    for requirement in requirements:
        if not requirement:
            continue

        if isinstance(requirement, str):
            lines.append(clean_html(requirement))
            continue

        title = requirement.get("title")
        if title:
            lines.append(clean_html(title))

        for child in requirement.get("children", []):
            text = clean_html(child)
            if text:
                lines.append(text)

    return lines


if __name__ == "__main__":
    projects = get_projects()
    print(f"老大，有{len(projects)}个项目")

    result = []

    for i, project in enumerate(projects, 1):
        id = project["proId"]
        url = API + "/getProDetail"
        response = requests.post(url, json={"programId": id}, timeout=20)
        detail = response.json()

        name = detail.get("programName") or project.get("programName")
        difficulty = detail.get("difficulty") or project.get("difficulty")
        tags = get_tag(detail)

        project_data = {
            "programCode": detail.get("programCode"),
            "proId": id,
            "organization": detail.get("orgName") or project.get("orgName"),
            "name": name,
            "difficulty": difficulty,
            "techTags": tags,
            "description": clean_html(detail.get("programDesc")),
            "outputRequirement": get_requirement(detail),
        }
        result.append(project_data)

        print(f"[{i}/{len(projects)}] {name} | {difficulty} | {'、'.join(tags)}")

    with open("ospp2025_projects_basic.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("老大爬完了喵")
