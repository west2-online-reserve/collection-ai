from enum import Enum


class ConditionType(Enum):
    REQUIRED_AFFINITY = "required_affinity"


class EffectType(Enum):
    MODIFY_AFFINITY = "modify_affinity"
