import requests
import asyncio

import aiohttp
from aiohttp_retry import ExponentialRetry, RetryClient
from asgiref.sync import async_to_sync

import json
import time

from tqdm import tqdm


def generate_payload(text, method, summary_type):
    if method == "zero_shot" and summary_type == "abstractive":
        prompt_text = f"""Your job is to produce summaries\n\nWrite a concise summary of the following:\n\n------------\n{text}\n------------\n"""
    elif method == "zero_shot" and summary_type == "extractive":
        prompt_text = f"""Your job is to produce summaries\n\nWrite a concise extractive summary of the following by selecting the key sentences that convey the gist of the given text and output them verbatim without any changes, paraphrasing, or rephrasing:\n\n------------\n{text}\n------------\n"""
    elif method == "few_shot" and summary_type == "extractive":
        prompt_text = f"""Your job is to produce summaries\n\nHere are a few training examples of the summarization task:\n------------\n"Text"\nChelsea are waiting on the fitness of John Terry ahead of Wednesday's Champions League match with Valencia, but Frank Lampard has been ruled out. John Terry tries out his protective mask during training for Chelsea on Tuesday. Center-back Terry suffered a broken cheekbone during Saturday's 0-0 draw with Fulham, and Chelsea manager Avram Grant will see how he fares during training on Tuesday before making a decision on his availability. Terry trained at Valencia's Mestalla stadium with a face mask on after surgery on Sunday. "John Terry wants to play which is very good. Now we need to wait for training and then we will speak with the medical department and decide," said Grant. Grant has confirmed that Lampard will definitely sit the game out though as the midfielder continues to recover from his thigh injury. Midfielder Michael Essien, who scored a last-minute winner for Chelsea to knock Valencia out of last season's Champions League, has also been battling a leg injury but he took part in training on Tuesday and is expected to play.\n"Summary"\nChelsea are still waiting on the fitness of John Terry ahead of the Champions League match with Valencia. Frank Lampard will definitely sit the game out though as the midfielder continues to recover from his thigh injury. Michael Essien has also been battling a leg injury but he took part in training on Tuesday and is expected to play.\n------------\n\nNow, write a concise extractive summary of the following by selecting the key sentences that convey the gist of the given text and output them verbatim without any changes, paraphrasing, or rephrasing:\n\n------------\n{text}\n------------\n"""
    elif method == "few_shot" and summary_type == "abstractive":
        prompt_text = f"""Your job is to produce summaries\n\nHere are a few training examples of the summarization task:\n------------\n"Text"\nVice President Dick Cheney will serve as acting president briefly Saturday while President Bush is anesthetized for a routine colonoscopy, White House spokesman Tony Snow said Friday. Bush is scheduled to have the medical procedure, expected to take about 2 1/2 hours, at the presidential retreat at Camp David, Maryland, Snow said. Bush's last colonoscopy was in June 2002, and no abnormalities were found, Snow said. The president's doctor had recommended a repeat procedure in about five years. The procedure will be supervised by Dr. Richard Tubb and conducted by a multidisciplinary team from the National Naval Medical Center in Bethesda, Maryland, Snow said. A colonoscopy is the most sensitive test for colon cancer, rectal cancer and polyps, small clumps of cells that can become cancerous, according to the Mayo Clinic. Small polyps may be removed during the procedure. Snow said that was the case when Bush had colonoscopies before becoming president. Snow himself is undergoing chemotherapy for cancer that began in his colon and spread to his liver. Snow told reporters he had a chemo session scheduled later Friday. Watch Snow talk about Bush's procedure and his own colon cancer. "The president wants to encourage everybody to use surveillance," Snow said. The American Cancer Society recommends that people without high-risk factors or symptoms begin getting screened for signs of colorectal cancer at age 50.\n"Summary"\nPresident Bush will have a routine colonoscopy Saturday. While he's anesthetized, his powers will be transferred to the vice president. Bush had last colonoscopy in 2002, which found no problems.\n------------\n\nNow, write a concise summary of the following:\n\n------------\n{text}\n------------\n"""

    return {
        "model": {
            "id": "openchat_v3.2_mistral",
            "name": "OpenChat Aura",
            "maxLength": 30000,
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
        "temperature": 0.2,
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
async def get_summaries(papers, method, summary_type):
    generated_summaries = []
    for paper in tqdm(papers):
        result_dict = {}

        result_dict["title"] = paper["title"]
        result_dict["gt_summary"] = paper["summary"]
        result_dict["id"] = paper["id"]

        sectionwise_summaries = [None] * len(paper["document"])
        tasks = [
            asyncio.ensure_future(
                generate_summary(
                    "https://openchat.team/api/chat",
                    generate_payload(section["text"], method, summary_type),
                    sectionwise_summaries,
                    idx,
                )
            )
            for idx, section in enumerate(paper["document"])
        ]

        await asyncio.gather(*tasks)

        result_dict["pred_summary"] = requests.post(
            "https://openchat.team/api/chat",
            json=generate_payload(
                "\n".join(sectionwise_summaries), method, summary_type
            ),
            headers={"Content-Type": "application/json"},
        ).text

        result_dict["extraction_type"] = summary_type
        generated_summaries.append(result_dict)

        time.sleep(1)

    return generated_summaries


if __name__ == "__main__":
    with open("papers.json", "r", encoding="utf-16") as file:
        papers = json.load(file)

    for method, summary_type in [
        ("few_shot", "extractive"),
        ("zero_shot", "extractive"),
        ("few_shot", "abstractive"),
        ("zero_shot", "abstractive"),
    ]:
        print(f"Generating summaries for '{method}-{summary_type}'")
        generated_summaries = get_summaries(papers, method, summary_type)
        with open(
            f"{method}_generated_summaries_{summary_type}.json", "w", encoding="utf-16"
        ) as file:
            json.dump(generated_summaries, file, indent=4)
        print(f"Summaries generated'")
        print("--------------------------")
