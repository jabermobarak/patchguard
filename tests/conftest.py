import pytest

from patchguard.data import demo_splits
from patchguard.model import Detector


@pytest.fixture(scope="session")
def trained():
    train, cal, _ = demo_splits()
    return Detector.fit(train, cal)
