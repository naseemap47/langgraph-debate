import pytest
from debate.drivers.adapters import get_adapter
from debate.drivers.browser import BrowserCDPDriver

def test_get_adapter_exists_for_all_models():
    for name in ["claude", "chatgpt", "gemini", "deepseek"]:
        adapter = get_adapter(name)
        assert adapter is not None
        assert adapter.name == name
        assert adapter.base_url.startswith("https://")

def test_get_adapter_unknown_raises():
    with pytest.raises(ValueError):
        get_adapter("unknown_model")

def test_browser_driver_init():
    driver = BrowserCDPDriver("claude", cdp_url="http://localhost:9222")
    assert driver.model_name == "claude"
    assert driver.cdp_url == "http://localhost:9222"
