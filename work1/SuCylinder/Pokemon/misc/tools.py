from time import sleep
from settings import SLEEP_TIME


def delay(func):
    def withdelay(*a, **b):
        func(*a, **b)
        sleep(SLEEP_TIME)
        return

    return withdelay


printWithDelay = delay(print)
