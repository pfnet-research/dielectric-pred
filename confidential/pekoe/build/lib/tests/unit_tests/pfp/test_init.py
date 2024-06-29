import pkg_resources
import pytest

import pfp


def test_version():
    expect = pkg_resources.get_distribution("pfp-base").version
    actual = pfp.__version__
    assert expect == actual


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
