import pytest
from unittest.mock import patch, MagicMock
from debate.utils.cdp_check import check_cdp_status

def test_check_cdp_status_offline():
    status = check_cdp_status("http://localhost:59999")
    assert status["connected"] is False
    assert "error" in status

@patch("urllib.request.urlopen")
def test_check_cdp_status_online(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'[{"url": "https://claude.ai/chat/1"}, {"url": "https://chatgpt.com/"}]'
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    status = check_cdp_status("http://localhost:9222")
    assert status["connected"] is True
    assert status["tabs"]["claude"] is True
    assert status["tabs"]["chatgpt"] is True
    assert status["tabs"]["gemini"] is False
    assert status["tabs"]["deepseek"] is False
