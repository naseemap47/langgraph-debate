from playwright.async_api import Page

class DeepSeekAdapter:
    name = "deepseek"
    base_url = "https://chat.deepseek.com/"
    match_url_pattern = "deepseek.com"

    async def submit_prompt(self, page: Page, prompt: str):
        input_elem = await page.wait_for_selector('textarea', timeout=15000)
        await input_elem.click()
        await page.keyboard.insert_text(prompt)
        await page.wait_for_timeout(300)
        await page.keyboard.press("Enter")

    async def wait_for_completion(self, page: Page, timeout_seconds: int = 120) -> str:
        await page.wait_for_timeout(1000)
        try:
            await page.wait_for_selector('.ds-icon-stop, button[aria-label*="Stop"]', state="detached", timeout=timeout_seconds * 1000)
        except Exception:
            pass
        msgs = await page.query_selector_all('.ds-markdown, div[class*="message-assistant"]')
        if msgs:
            return (await msgs[-1].inner_text()).strip()
        return "DeepSeek finished responding."
