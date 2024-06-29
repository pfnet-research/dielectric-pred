from invoke.config import merge_dicts


def run(c, command, **kwargs):
    default_config = {
        "echo": True,
        "warn": False,
    }
    return c.run(command, **merge_dicts(default_config, kwargs))
