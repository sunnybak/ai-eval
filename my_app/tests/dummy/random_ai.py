from openai import AsyncOpenAIClient


async def run_app(_, model, len):
    res = generator()
    await asyncio.sleep(random.random())
    res = await a_openai_call('write a 200 word essay on happiness')
    return [scorer_1(res), scorer_2(res)]


def test_dummy():