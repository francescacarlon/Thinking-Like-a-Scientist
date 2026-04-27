from openai import AsyncOpenAI

from .prompts import SYSTEM_PROMPT


async def async_make_openai_request(client: AsyncOpenAI, user_prompt: str):
    return await client.responses.create(
        model="gpt-5.1-2025-11-13",
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
        reasoning={"effort": "medium"},
        # tools=[{"type": "web_search"}],
    )
