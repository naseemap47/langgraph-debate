import asyncio
from playwright.async_api import async_playwright
from debate.drivers.base import BaseModelDriver
from debate.drivers.adapters import get_adapter

class BrowserCDPDriver(BaseModelDriver):
    def __init__(self, model_name: str, cdp_url: str = "http://localhost:9222"):
        self.model_name = model_name.lower().strip()
        self.cdp_url = cdp_url
        self.adapter = get_adapter(self.model_name)
        self._lock = asyncio.Lock()

    async def _get_or_create_page(self, browser_context):
        for page in browser_context.pages:
            if self.adapter.match_url_pattern in page.url.lower():
                await page.bring_to_front()
                return page
        page = await browser_context.new_page()
        await page.goto(self.adapter.base_url, timeout=30000, wait_until="domcontentloaded")
        return page

    async def send_prompt(self, prompt: str, new_chat: bool = True) -> str:
        async with self._lock:
            async with async_playwright() as p:
                try:
                    browser = await p.chromium.connect_over_cdp(self.cdp_url)
                except Exception as e:
                    raise ConnectionError(
                        f"Cannot connect to Chrome at {self.cdp_url}. "
                        f"Make sure Chrome was started with './scripts/run_chrome.sh'. Error: {e}"
                    )
                context = browser.contexts[0] if browser.contexts else await browser.new_context()
                page = await self._get_or_create_page(context)

                if new_chat and page.url.lower() != self.adapter.base_url.lower():
                    await page.goto(self.adapter.base_url, timeout=30000, wait_until="domcontentloaded")

                await self.adapter.submit_prompt(page, prompt)
                response = await self.adapter.wait_for_completion(page)
                return response

    async def health_check(self) -> bool:
        async with async_playwright() as p:
            try:
                await p.chromium.connect_over_cdp(self.cdp_url)
                return True
            except Exception:
                return False
