import requests
import json
import jsonpath
import re

url = "https://summer-ospp.ac.cn/api/getProList"
session = requests.session()
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36 Edg/146.0.0.0"
}
session.post(url, headers=headers)
num = 0
with open("ospp.csv", "w", encoding="utf-8") as f:
    for i in range(1, 13):
        data = {
            "difficulty": [],
            "lang": "zh",
            "orgName": [],
            "pageNum": i,
            "pageSize": "50",
            "programName": "",
            "programmingLanguageTag": [],
            "supportLanguage": [],
            "techTag": [],
        }
        text = session.post(url, data=data).json()
        programName = jsonpath.jsonpath(text, "$..programName")  # 获取项目名称
        difficulty = jsonpath.jsonpath(text, "$..difficulty")  # 获取项目难度
        techTag = jsonpath.jsonpath(text, "$..techTag")  # 获取项目技术标签
        programCode = jsonpath.jsonpath(text, "$..programCode")  # 获取项目代码
        for i in range(
            len(programCode)
        ):  # 获取每个项目的具体信息，包括项目简述、项目产出要求
            data = {"programId": programCode[i], "type": "org"}
            text = session.post(
                "https://summer-ospp.ac.cn/api/getProDetail", data=data
            ).json()
            programDesc = re.sub(
                r"<(.*?)>", "", jsonpath.jsonpath(text, "$..programDesc")[0]
            )
            outputRequirement = jsonpath.jsonpath(text, "$..outputRequirement..title")
            techRequirement = jsonpath.jsonpath(text, "$..techRequirement..title")
            orgProgramId = jsonpath.jsonpath(text, "$..orgProgramId")[0]
            # pdf={
            #     "proId":orgProgramId
            # }
            # response=session.post("https://summer-ospp.ac.cn/api/publicApplication",data=pdf)
            # with open(f"{re.sub(r'[\\/:*?"<>|]', '_', programName[i])}项目申请书.pdf","wb") as pf:
            #     pf.write(response.content)
            # 用于下载项目申请书
            num += 1
            f.write(f"{num}. 项目名:{programName[i]}\n")
            f.write(f"难度：{difficulty[i]}\n")
            f.write(f"技术标签：{techTag[i]}\n")
            f.write(f"项目简述：{programDesc}\n")
            f.write("项目产出要求：\n")
            for j in range(len(outputRequirement)):
                f.write(f"{outputRequirement[j]}\n")
            f.write("技术要求：\n")
            for j in range(len(techRequirement)):
                f.write(f"{techRequirement[j]}\n")
            f.write(
                "------------------------------------------------------------------------------------------\n"
            )
