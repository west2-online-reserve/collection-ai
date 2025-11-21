
from typing import Dict
from pathlib import Path
import csv
import re

from bs4 import BeautifulSoup
import requests

from program import Program


TOTAL_PAGES = 12
ROOT_DIR = Path(__file__).absolute().parent


class Summer:
    def __init__(self):
        self.count: int = 0

        self.file = open(ROOT_DIR/"data.csv", "w", encoding = "utf-8", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["项目编号","项目名称","项目难度","技术标签","项目简述","产出要求"])


    def get_programs(self,page):  
        print(f"Posting the {page} page: ")
        data = {"programName":"",
                "pageNum":page,
                "pageSize":50,
                "lang":"zh"}

        response = requests.post("https://summer-ospp.ac.cn/api/getProList", data=data)
        response.encoding = "utf-8"

        for program_dict in response.json()['rows']:

            program = Program()
            program.programCode = program_dict['programCode']
            program.programName = program_dict['programName']
            program.difficulty = program_dict['difficulty']
            program.orgId = program_dict['orgId']
            program.proId = program_dict['proId']

            techTags = re.findall(r",\"([^\[\],]{1,})\"]",program_dict['techTag'])
            for techTag in techTags:
                program.techTag += f"#{techTag} "

            self.count += 1
            self.programs.append(program)
            self.get_detail(program)


    def get_detail(self,program: Program):
        print(f"Posting the {self.count} program...")

        data = {"programId": program.programCode,
                "type": "org"}
        
        response = requests.post("https://summer-ospp.ac.cn/api/getProDetail", data=data)
        response.encoding = 'utf-8'

        detail_dict = response.json()

        for outputRequirement in detail_dict['outputRequirement'][1:]:
            program.outputRequirement.append(outputRequirement['title'])

        programDesc_soup = BeautifulSoup(detail_dict['programDesc'], 'lxml')
        program.programDesc = programDesc_soup.get_text()

        self.writer.writerow(program.information())


    def main(self):
        for page in range(TOTAL_PAGES):
            self.get_programs(page+1)
        
        print("Completed.")
        self.file.close()


if __name__ == "__main__":
    summer = Summer()
    summer.main()