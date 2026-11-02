import pytest
from debate.drivers.base import BaseModelDriver

class DummyDriver(BaseModelDriver):
    def __init__(self, name: str = "dummy"):
        self.model_name = name

    async def send_prompt(self, prompt: str, new_chat: bool = True) -> str:
        return f"Response to: {prompt}"

    async def health_check(self) -> bool:
        return True

@pytest.mark.asyncio
async def test_dummy_driver_implements_interface():
    driver = DummyDriver()
    assert driver.model_name == "dummy"
    assert await driver.health_check() is True
    res = await driver.send_prompt("Hello")
    assert res == "Response to: Hello"

def test_cannot_instantiate_abstract_driver():
    with pytest.raises(TypeError):
        BaseModelDriver()
