class SenPai:
    def __init__(self, name: str) -> None:
        self._lover = "None"
        self._name = name
        self.loved = False

    @property
    def lover(self):
        return self._lover

    @lover.setter
    def lover(self, name: str):
        self.loved = True
        self._lover = name

    @property
    def name(self) -> str:
        return self._name if not self.loved else "I_love_" + self.lover


if __name__ == "__main__":
    sen_pai_list: list[SenPai] = []

    sen_pai_num = int(input())

    for i in range(sen_pai_num):
        sen_pai_list.append(SenPai(input()))

    love_num = int(input())

    for i in range(love_num):
        love_str = input().split()
        lover_index = int(love_str[0]) - 1
        loved_index = int(love_str[1]) - 1

        lover = sen_pai_list[lover_index]
        loved = sen_pai_list[loved_index]

        lover.lover = loved.name

        sen_pai_list[lover_index] = lover

    print(sen_pai_list[0].name)
