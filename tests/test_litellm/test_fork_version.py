from importlib.metadata import version


def test_fork_version_distinguishes_upstream_release():
    assert version("litellm") == "1.92.0+hemuling.1"
