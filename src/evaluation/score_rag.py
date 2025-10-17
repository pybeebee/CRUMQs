import os
import json
import re
import string
import argparse
import json_repair
from collections import Counter
from tqdm import tqdm

from prompts_evaluation import out_of_database_check_prompt, unanswerable_check_prompt
from src.crumqs_generation.utils_inference import *

from google.generativeai import GenerationConfig
import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash-lite")

LLM_EVAL_PROMPT = """Determine whether the predicted answer contains text semantically equivalent to the ground truth. Output ONLY True or False.
ground truth options = {target}
predicted answer = {pred}"""

### HELPER: Normalize Answer Text
def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

### HELPER: Score Accuracy
def llm_eval(target, pred):
    
    # Format prompt
    prompt = LLM_EVAL_PROMPT.format(target=target, pred=pred)

    # Get response
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY")) 
        model_acc = genai.GenerativeModel("gemini-2.0-flash-lite")
        generation_config = GenerationConfig( 
            max_output_tokens=2,
            candidate_count=1,
        )
        output = model_acc.generate_content(
            prompt,
            generation_config=generation_config,
            ).text.strip()
        response = normalize_answer(output)
        
    except Exception as e: 
        response = ""
        print("llm scoring error:", e)
    
    # 1 if True, 0 o.w.
    score = 1. if "true" in response else 0. 
    # print(score)

    return score

### HELPER: Evaluate if Answer is Acceptable (Single)
def eval_acceptable(question, answer):

    prompt = out_of_database_check_prompt.format(request=question, response=answer).prompt_str

    try:
        outputs = model.generate_content(
            prompt,
            generation_config=GenerationConfig( 
                max_output_tokens=500, 
                candidate_count=1,
                top_k=1,
                temperature=0,
            ),
        )
        output = outputs.text.strip()
        response = json_repair.repair_json(output, return_objects=True)

        acceptable, reason_acceptable = response["verdict"], response["reason"]
    except Exception as e:
        print(e)
        acceptable, reason_acceptable = None, None

    return acceptable, reason_acceptable

### HELPER: Evalute if Question is Deemed UNAnswerable (Single)
def eval_unanswerable(question, answer, generator_llm_model=None):

    ### CHECK: Does model indicate q is unanswerable?
    prompt = unanswerable_check_prompt.format(question=question, answer=answer).prompt_str
    
    try:
        outputs = model.generate_content(
            prompt,
            generation_config=GenerationConfig( 
                max_output_tokens=500, 
                candidate_count=1,
                top_k=1,
                temperature=0,
            ),
        )
        output = outputs.text.strip()
        response = json_repair.repair_json(output, return_objects=True)
        answerable, reason_unanswerable = response["verdict"], response["reason"]
    except Exception as e:
        print(e)
        answerable, reason_unanswerable = None, None

    return answerable, reason_unanswerable 

### HELPER: Evaluate Answerability + Acceptability for ALL Samples
def evaluate_dataset(questions, answers):

    answerable_list, reason_unanswerable_list = [], [] 
    acceptable_list, reason_acceptable_list = [], [] 

    assert(len(questions)==len(answers))
    for q, a in tqdm(zip(questions, answers), total=len(questions)):

        ### Judge Answerability
        answerable, reason_unanswerable = eval_unanswerable(
            question=q, 
            answer=a, 
        )
        if answerable:
            answerable_list.append(answerable)
            reason_unanswerable_list.append(reason_unanswerable)
        else:
            answerable_list.append(None)
            reason_unanswerable_list.append(None)

        ### Judge Acceptability
        acceptable, reason_acceptable = eval_acceptable(
            question=q, 
            answer=a,
        )
        if acceptable:
            acceptable_list.append(acceptable)
            reason_acceptable_list.append(reason_acceptable)
        else:
            acceptable_list.append(None)
            reason_acceptable_list.append(None)

    return answerable_list, reason_unanswerable_list, acceptable_list, reason_acceptable_list

### HELPER: Compute Unanswerable Ratio + Clarification N Ratio
def score_dataset(answerable_list, acceptable_list, orig_ans=None, orig_acc=None):

    ### Score Answerability
    count_ans = Counter(answerable_list)
    if not orig_ans:
        total_ans = sum([1 for answerable in answerable_list if answerable in ["1", "-1", "0"]])
    else: 
        total_ans = sum([1 for answerable in orig_ans if answerable in ["1", "-1", "0"]])
    answered_ratio = count_ans["1"] / total_ans
    unanswerable_ratio = count_ans["-1"] / total_ans
    clarification_needed_ratio = (count_ans["0"]) / total_ans

    ### Score Acceptability
    count_acc = Counter(acceptable_list)
    if not orig_acc:
        total_acc = sum([1 for acceptable in acceptable_list if acceptable in ["1", "-1"]])
    else:
        total_acc = sum([1 for acceptable in orig_acc if acceptable in ["1", "-1"]])
    acceptable_ratio = count_acc["1"] / total_acc
    unacceptable_ratio = count_acc["-1"] / total_acc

    return answered_ratio, unanswerable_ratio, clarification_needed_ratio, acceptable_ratio, unacceptable_ratio

### MAIN FUNCTION
def main(args):

    # Load Q&A's
    with open(args.results_path, "r") as f:
        data = json.load(f)
    questions = data["questions"]
    answers = data["answers"]

    if args.score_accuracy_only: 

        ### Load Ground Truth Answers
        with open(args.gt_data_path, "r") as f:
            gt_data = json.load(f)

        ### Score Accuracy with LLM_Eval
        acc_scores = []
        for q, a in tqdm(zip(questions, answers), total=len(questions)):
            gt_idx = gt_data['question'].index(q)
            gt_a = gt_data['ground_truth'][gt_idx] 
            acc_score = llm_eval(target=gt_a, pred=a)
            acc_scores.append(acc_score)
        
        accuracy = sum(acc_scores) / len(acc_scores)

        ### Save Results
        save_path =  os.path.join(os.path.dirname(args.results_path), "scores_acc_only.json")
        results = {
            "acc": accuracy,
            "acc_judgments": acc_scores
        }
        print("final acc:", accuracy)

    elif args.score_by_num_hops:

        ### Load Ground Truth Answers
        with open(args.gt_data_path, "r") as f:
            gt_data = json.load(f)

        ### Load Existing Per-Question Scores
        with open(args.results_path.replace("predictions.", "scores."), "r") as f:
            scores = json.load(f)
        with open(args.results_path.replace("predictions.", "scores_acc_only."), "r") as f:
            acc_scores = json.load(f)

        answerable_list = scores['answerable_list']
        acceptable_list = scores['acceptable_list']
        acc_scores_list = acc_scores['acc_judgments']

        q_to_idx = {q: i for i, q in enumerate(scores["questions"])}

        scores_by_hop = {}
        for cur_num_hops in set(gt_data["num_hops"]):
            indices = [
                q_to_idx[q] 
                for i, q in enumerate(gt_data["question"]) 
                if gt_data["num_hops"][i] == cur_num_hops and q in q_to_idx
            ]

            answerable_subset = [answerable_list[i] for i in indices]
            acceptable_subset = [acceptable_list[i] for i in indices]
            acc_subset = [acc_scores_list[i] for i in indices]

            answered_ratio, unanswerable_ratio, clarification_needed_ratio, acceptable_ratio, unacceptable_ratio = score_dataset(
                answerable_subset,
                acceptable_subset,
                # orig_ans=answerable_list,
                # orig_acc=acceptable_list,
            )
            accuracy = sum(acc_subset) / len(answerable_list)

            scores_by_hop[str(cur_num_hops)] = {
                "answered_ratio": answered_ratio,
                "unanswerable_ratio": unanswerable_ratio, "clarification_needed_ratio": clarification_needed_ratio,
                "acceptable_ratio": acceptable_ratio,
                "unacceptable_ratio": unacceptable_ratio,
                "accuracy": accuracy,
            }
        
        save_path = os.path.join(args.results_path.replace("predictions.json", "scores_by_hop.json"))
        results = scores_by_hop

    else: 
        ### Judge Results
        answerable_list, reason_unanswerable_list, acceptable_list, reason_acceptable_list = evaluate_dataset(
            questions=questions, 
            answers=answers,
        )

        ### Score Results
        answered_ratio, unanswerable_ratio, clarification_needed_ratio, acceptable_ratio, unacceptable_ratio = score_dataset(answerable_list, acceptable_list)

        ### Save Results
        save_path = os.path.join(os.path.dirname(args.results_path), "scores.json")
        results = {
            "answered_ratio": answered_ratio,
            "unanswerable_ratio": unanswerable_ratio, "clarification_needed_ratio": clarification_needed_ratio,
            "acceptable_ratio": acceptable_ratio,
            "unacceptable_ratio": unacceptable_ratio,
            "questions": questions,
            "answers": answers,
            "answerable_list": answerable_list,
            "reason_unanswerable_list": reason_unanswerable_list,
            "acceptable_list": acceptable_list,
            "reason_acceptable_list": reason_acceptable_list,
        }
        print(answered_ratio, unanswerable_ratio, clarification_needed_ratio)

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gt_data_path",
        type=str,
    )
    parser.add_argument(
        "--results_path",
        type=str,
    )
    parser.add_argument(
        "--score_accuracy_only",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--score_by_num_hops",
        action="store_true",
        default=False
    )

    args = parser.parse_args()

    main(args)
