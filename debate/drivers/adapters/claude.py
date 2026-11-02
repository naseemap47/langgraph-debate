from playwright.async_api import Page

class ClaudeAdapter:
    name = "claude"
    base_url = "https://claude.ai/new"
    match_url_pattern = "claude.ai"

    async def submit_prompt(self, page: Page, prompt: str):
        input_elem = await page.wait_for_selector('div[contenteditable="true"]', timeout=15000)
        await input_elem.click()
        await page.keyboard.insert_text(prompt)
        await page.wait_for_timeout(300)
        await page.keyboard.press("Enter")

    async def wait_for_completion(self, page: Page, timeout_seconds: int = 120) -> str:
        await page.wait_for_timeout(1000)
        try:
            stop_btn = await page.query_selector('button[aria-label*="Stop"]')
            if stop_btn:
                await page.wait_for_selector('button[aria-label*="Stop"]', state="detached", timeout=timeout_seconds * 1000)
        except Exception:
            pass
        msgs = await page.query_selector_all('div.font-claude-message, div[data-is-streaming="false"]')
        if msgs:
            return (await msgs[-1].inner_text()).strip()
        all_blocks = await page.query_selector_all('div.prose, div[class*="standard-markdown"]')
        if all_blocks:
            return (await all_blocks[-1].inner_text()).strip()
        return "Claude finished responding."
