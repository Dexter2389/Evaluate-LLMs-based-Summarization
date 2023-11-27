import asyncio

import aiohttp
from aiohttp_retry import ExponentialRetry, RetryClient
from asgiref.sync import async_to_sync

import json

from tqdm import tqdm


def generate_payload(gt_text, pred_text):
    prompt_text = f"""Your job is to generate a score between two summaries based on their quality and similarity in meaning.\n\nWe have provided a ground_truth summary of a research paper: {gt_text}\nWe have the opportunity to evaluate a predicted summary as provided below.\n------------\n{pred_text}\n------------\nGiven the ground_truth summary, evaluate and score the predicted summary on a scale of 0.00 to 1.00\nIf the ground_truth summary and predicted summary are exactly the same give them a score of 1.00 but if they are completely different give them a score of 0.00"""

    return {
        "model": {
            "id": "openchat_v3.2_mistral",
            "name": "OpenChat Aura",
            "maxLength": 24000,
            "tokenLimit": 8192,
        },
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        "key": "",
        "prompt": " ",
        "temperature": 0.1,
    }


async def generate_summary(url, payload, response_list, request_index):
    retry_options = ExponentialRetry(
        attempts=5, statuses=[500], exceptions={aiohttp.ClientConnectorError}
    )
    async with RetryClient(
        retry_options=retry_options, raise_for_status=False
    ) as client:
        async with client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            if response.status == 200:
                try:
                    response_list[request_index] = await response.text()
                except Exception as e:
                    print(e)
            else:
                print(response.status)


@async_to_sync
async def get_summary_evals(papers):
    evaluate_summaries = [None] * len(papers)
    tasks = [
        asyncio.ensure_future(
            generate_summary(
                "https://openchat.team/api/chat",
                generate_payload(paper["gt_summary"], paper["pred_summary"]),
                evaluate_summaries,
                idx,
            )
        )
        for idx, paper in enumerate(papers)
    ]

    await asyncio.gather(*tasks)
    return evaluate_summaries


for method, summary_type in [
    ("few_shot", "extractive"),
    ("zero_shot", "extractive"),
    ("few_shot", "abstractive"),
    ("zero_shot", "abstractive"),
    ("chain_of_thought", "none"),
]:
    with open(
        f"{method}_generated_summaries_{summary_type}.json", "r", encoding="utf-16"
    ) as file:
        papers = json.load(file)

    evaluated_summaries = get_summary_evals(papers)
    print(f"'{method}-{summary_type}': {evaluated_summaries}")
