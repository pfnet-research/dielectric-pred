import pkg_resources
import pytest

import pekoe


def test_version():
    expect = pkg_resources.get_distribution("pekoe").version
    actual = pekoe.__version__
    assert expect == actual


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
