
# # Default
import os
import asyncio

from groq import AsyncGroq
from openai import AsyncOpenAI
client = AsyncGroq(
    # This is the default and can be omitted
    api_key='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',
)

# client = AsyncOpenAI(
#     api_key='sk-proj-2UDNk6vwtcfFuabLNGVFT3BlbkFJor4tAlnwexn6RgCFKoBc',
# )

from ai_eval.scorers.chat.chat_scorer import ChatScorer


async def run():

    time_start = asyncio.get_event_loop().time()
    async def run_chat():
        messages = [
            {
                "role": "system",
                "content": "talk to yourself about life",
            },
        ]
        while len(messages) < 3:
            stream = await client.chat.completions.create(
                messages=messages,
                model="llama3-8b-8192",
                # model="gpt-4o",
                stream=True,
            )
            msg = ""
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="")
                    msg += content
            # msg = chat_completion.choices[0].message.content
            messages.append({"role": "assistant", "content": msg})

        time_end = asyncio.get_event_loop().time()
        interval = time_end - time_start
        print(f"Elapsed time: {interval}")
        chat_scorer = ChatScorer(messages)
        counts = chat_scorer.chat_total_tokens()
        cost = 0.05e-6*counts['input'] + 0.08e-6*counts['output']
        cost_per_token = cost / (counts['input'] + counts['output'])
        total_count = counts['input'] + counts['output']
        
        print(f"Cost: {cost}")
        print(f"TPS:", total_count/interval)
        return cost, total_count, interval

    tasks = [run_chat() for _ in range(5)]
    results = await asyncio.gather(*tasks)

    # print the totals
    #total cost

    total_cost = sum([r[0] for r in results])
    avg_interval = sum([r[2] for r in results])/len(results)
    avg_tps = sum([r[1] for r in results]) / sum([r[2] for r in results])
    print(f"Total cost: {total_cost}")

    print(f'Total tokens: {sum([r[1] for r in results])}')
    print(f"Average TPS: {avg_tps}")
    print(f"Average interval: {avg_interval}")
asyncio.run(run())