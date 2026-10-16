from ..util.dialogue_helper import Choice, Dialogue


class Character:
    id: str = "none"
    name: str = "none"

    @classmethod
    def talk(cls, content: str) -> Dialogue:
        return Dialogue(cls.name, content)

    @classmethod
    def choice(cls, content: str, group_to: str) -> Choice:
        return Choice(cls.id, content, group_to)
