from playwright.async_api import Page

class ChatGPTAdapter:
    name = "chatgpt"
    base_url = "https://chatgpt.com/"
    match_url_pattern = "chatgpt.com"

    async def submit_prompt(self, page: Page, prompt: str):
        input_elem = await page.wait_for_selector('#prompt-textarea, div[contenteditable="true"]', timeout=15000)
        await input_elem.click()
        await page.keyboard.insert_text(prompt)
        await page.wait_for_timeout(300)
        await page.keyboard.press("Enter")

    async def wait_for_completion(self, page: Page, timeout_seconds: int = 120) -> str:
        await page.wait_for_timeout(1000)
        try:
            await page.wait_for_selector('button[data-testid="stop-button"]', state="detached", timeout=timeout_seconds * 1000)
        except Exception:
            pass
        msgs = await page.query_selector_all('[data-message-author-role="assistant"]')
        if msgs:
            return (await msgs[-1].inner_text()).strip()
        return "ChatGPT finished responding."
