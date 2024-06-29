import invoke

ns = invoke.Collection()

try:
    import tasks.docker

    ns.add_collection(tasks.docker)
except ImportError:
    pass

try:
    import tasks.deploy

    ns.add_collection(tasks.deploy)
except ImportError:
    pass

try:
    import tasks.lint

    ns.add_collection(tasks.lint)
except ImportError:
    pass

try:
    import tasks.lock

    ns.add_collection(tasks.lock)
except ImportError:
    pass
