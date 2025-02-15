import pytest
from logfire import configure

@pytest.fixture(autouse=True)
def disable_logfire():
    configure(
        local=True,
        send_to_logfire=False
    )