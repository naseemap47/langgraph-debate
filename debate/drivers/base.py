from abc import ABC, abstractmethod

class BaseModelDriver(ABC):
    model_name: str

    @abstractmethod
    async def send_prompt(self, prompt: str, new_chat: bool = True) -> str:
        """Sends a prompt to the model and returns its full response."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Verifies session is active and responsive."""
        pass
