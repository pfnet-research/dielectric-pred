from enum import IntEnum
from typing import Tuple


class MessageEnum(IntEnum):
    ExperimentalElementWarning = 1001
    UnexpectedElementWarning = 1002

    def __str__(self) -> str:
        message_type, message_str = message_info(self)
        if message_type == MessageType.CustomMessage:
            return "Runtime message (ID {:d}): {:s}".format(int(self), message_str)
        elif message_type == MessageType.CustomWarning:
            return "Runtime warning (ID {:d}): {:s}".format(int(self), message_str)
        else:
            raise ValueError("Unknown MessageType")


class MessageType(IntEnum):
    CustomWarning = 20
    CustomMessage = 10


_CW = MessageType.CustomWarning
_CM = MessageType.CustomMessage
_MESSAGE_DICT = {
    MessageEnum.ExperimentalElementWarning: (
        _CW,
        "Experimental element was detected.\
         You can suppress this message by Estimator.set_message_status().",
    ),
    MessageEnum.UnexpectedElementWarning: (
        _CW,
        "Unexpected element was detected. There might be strange behavior.",
    ),
}


def message_info(message: MessageEnum) -> Tuple[MessageType, str]:
    return _MESSAGE_DICT[message]
