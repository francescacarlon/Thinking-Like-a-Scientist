import json


def load_research_questions(jsonl_file):
    questions = []
    
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            custom_id = row.get("custom_id")

            try:
                message_block = row["response"]["body"]["output"][1]
                text = message_block["content"][0]["text"]

                parsed = json.loads(text)
                rq = parsed["research_question"]

                questions.append((custom_id, rq))

            except Exception as e:
                print(f"Could not parse row {custom_id}: {e}")

    return questions
