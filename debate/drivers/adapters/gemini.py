from playwright.async_api import Page

class GeminiAdapter:
    name = "gemini"
    base_url = "https://gemini.google.com/app"
    match_url_pattern = "gemini.google.com"

    async def submit_prompt(self, page: Page, prompt: str):
        input_elem = await page.wait_for_selector('rich-textarea div[contenteditable="true"], div.input-area', timeout=15000)
        await input_elem.click()
        await page.keyboard.insert_text(prompt)
        await page.wait_for_timeout(300)
        await page.keyboard.press("Enter")

    async def wait_for_completion(self, page: Page, timeout_seconds: int = 120) -> str:
        await page.wait_for_timeout(1500)
        try:
            await page.wait_for_selector('.sparkle-loading, .loading-indicator', state="detached", timeout=timeout_seconds * 1000)
        except Exception:
            pass
        msgs = await page.query_selector_all('model-response, message-content, .model-response-text')
        if msgs:
            return (await msgs[-1].inner_text()).strip()
        return "Gemini finished responding."
