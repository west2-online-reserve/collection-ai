class Program:
    def __init__(self):
        self.programCode: str = ""
        self.programName: str = ""

        self.difficulty: str = ""
        self.techTag:str = ""

        self.orgId: str = ""
        self.proId:str = ""

        self.programDesc: str = ""
        self.outputRequirement: str = []


    def information(self):
        return [self.programCode,self.programName,self.difficulty,\
            self.techTag,self.programDesc] + self.outputRequirement