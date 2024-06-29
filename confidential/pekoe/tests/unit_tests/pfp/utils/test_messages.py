from pfp.utils.messages import MessageEnum, MessageType, message_info


def test_message_info():
    for m in MessageEnum:
        message_type, message_str = message_info(m)
        assert isinstance(message_type, MessageType)
        assert isinstance(message_str, str)
