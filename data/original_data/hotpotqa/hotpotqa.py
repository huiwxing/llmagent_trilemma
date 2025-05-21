import json
from datasets import Dataset, DatasetDict

def load():

    with open(
        "data/hotpotqa/hotpot_dev_distractor_v1.json",
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)
        raw_data = []
        # 只测前500条
        for item in data[:500]:
            question = item["question"]
            answer = item["answer"]
            sf = item["supporting_facts"]
            context = item["context"]
            titles = [ele[0] for ele in sf]
            sentence_id = [ele[1] for ele in sf]
            result_sf = {"title": titles, "sen_id": sentence_id}
            titles = [ele[0] for ele in context]
            sentences = [ele[1] for ele in context]
            result_context = {"title": titles, "sentences": sentences}
            raw_data.append(
                {
                    "question": question,
                    "answer": answer,
                    "supporting_facts": result_sf,
                    "context": result_context,
                }
            )
    ds = Dataset.from_list(raw_data)
    return ds

import re
def general_postprocess(text: str) -> str:
    # Cut off the first newline, period, or comma
    truncated_text = re.split(r'[\n.,]', text, 1)[0]

    # Remove punctuation
    no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

    # Remove article
    no_articles = re.sub(r'\b(a|an|the)\b',
                         '',
                         no_punctuation,
                         flags=re.IGNORECASE)

    # Remove duplicated blank spaces
    cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()

    return cleaned_text

def score(predictions, references):

    if len(predictions) != len(references):
        return {"error": "predictions and references have different " "length"}
    processed_predictions = []
    for prediction in predictions:
        prediction = prediction.strip().split("\n")[0].lower()
        if "answer is" in prediction:
            prediction = prediction.split("answer is")[-1]
        prediction = general_postprocess(prediction)
        processed_predictions.append(prediction)
    processed_answers = [[general_postprocess(i).lower()] for i in references]

    details = []
    cnt = 0
    F1_all = 0
    for pred, cand_ans in zip(processed_predictions, processed_answers):
        detail = {"pred": pred, "answer": cand_ans, "correct": False, "F1_score": 0}
        # EM
        # is_correct = any([cand == pred for cand in cand_ans])
        # EM in
        is_correct = any([cand in pred for cand in cand_ans])
        cnt += int(is_correct)
        detail["correct"] = is_correct

        prediction_chars = pred.split()
        reference_chars = cand_ans[0].split()
        common_tokens = set(prediction_chars) & set(reference_chars)
        if not common_tokens:
            F1_score = 0
        else:
            num_common_tokens = len(common_tokens)
            precision = num_common_tokens / len(prediction_chars)
            recall = num_common_tokens / len(reference_chars)
            if precision + recall == 0:
                F1_score = 0
            else:
                F1_score = 2 * (precision * recall) / (precision + recall)
        detail["F1_score"] = F1_score

        details.append(detail)
        F1_all += F1_score

    EM_score = cnt / len(predictions)
    F1_avg = F1_all / len(predictions)
    return {"EM score": EM_score, "F1 score": F1_avg, "details": details}