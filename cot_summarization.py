from multiprocessing.pool import Pool
import requests
import json
import time


def generate_payload(text, current_summary, step):
    if step == "chain_of_thought":
        prompt_text = f"""Your job is to produce a final summary\n\nWe have provided an existing summary up to a certain point: {current_summary}\nWe have the opportunity to refine the existing summary (only if needed) with some more context below.\n------------\n{text}\n------------\nGiven the new context, refine the original summary\nIf the context isn't useful, return the original summary."""
    elif step == "base_summary":
        prompt_text = f"""Your job is to produce summaries\n\nWrite a concise summary of the following:\n\n------------\n{text}\n------------\n"""
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
        "temperature": 0.3,
    }


def summarization_task(paper):
    previous_summary = ""
    for idx, doc in enumerate(paper["document"]):
        if idx == 0:
            previous_summary = requests.post(
                "https://openchat.team/api/chat",
                json=generate_payload(doc["text"], "", "base_summary"),
                headers={"Content-Type": "application/json"},
            ).text
        elif idx == len(paper["document"]) - 1:
            result_dict = {}

            result_dict["title"] = paper["title"]
            result_dict["gt_summary"] = paper["summary"]
            result_dict["id"] = paper["id"]
            result_dict["extraction_type"] = "chain_of_thought"
            result_dict["pred_summary"] = requests.post(
                "https://openchat.team/api/chat",
                json=generate_payload(
                    doc["text"], previous_summary, "chain_of_thought"
                ),
                headers={"Content-Type": "application/json"},
            ).text

            return result_dict
        else:
            previous_summary = requests.post(
                "https://openchat.team/api/chat",
                json=generate_payload(
                    doc["text"], previous_summary, "chain_of_thought"
                ),
                headers={"Content-Type": "application/json"},
            ).text

    time.sleep(1)


if __name__ == "__main__":
    with open("papers.json", "r", encoding="utf-16") as file:
        papers = json.load(file)
        cot_generated_summaries = []
        with Pool(8) as pool:
            result = pool.map_async(summarization_task, papers)
            for result in result.get():
                cot_generated_summaries.append(result)
        with open("cot_summaries.json", "w", encoding="utf-16") as file:
            json.dump(cot_generated_summaries, file, indent=4)
