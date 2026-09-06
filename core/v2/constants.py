from enum import IntEnum


class State(IntEnum):
    WRONG = -1
    UNKNOWN = 0
    RIGHT = 1
    CONFLICT = 2
