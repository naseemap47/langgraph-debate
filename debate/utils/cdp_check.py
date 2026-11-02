import json
import urllib.request
from typing import Dict, Any

def check_cdp_status(endpoint: str = "http://127.0.0.1:9222") -> Dict[str, Any]:
    json_url = f"{endpoint.rstrip('/')}/json/list"
    result: Dict[str, Any] = {
        "connected": False,
        "endpoint": endpoint,
        "tabs": {
            "claude": False,
            "chatgpt": False,
            "gemini": False,
            "deepseek": False,
        },
        "raw_tab_count": 0,
    }
    try:
        req = urllib.request.Request(json_url, headers={"User-Agent": "LLM-Debate/1.0"})
        with urllib.request.urlopen(req, timeout=1.5) as response:
            data = json.loads(response.read().decode("utf-8"))
            result["connected"] = True
            result["raw_tab_count"] = len(data)
            for item in data:
                url = item.get("url", "").lower()
                if "claude.ai" in url:
                    result["tabs"]["claude"] = True
                if "chatgpt.com" in url or "openai.com" in url:
                    result["tabs"]["chatgpt"] = True
                if "gemini.google.com" in url:
                    result["tabs"]["gemini"] = True
                if "deepseek.com" in url:
                    result["tabs"]["deepseek"] = True
    except Exception as e:
        result["connected"] = False
        result["error"] = str(e)
    return result
