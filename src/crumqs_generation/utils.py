import re
from termcolor import colored

import json_repair

from src.crumqs_generation.prompts.verification import *

def parse_qa_outputs(output_strings):
        """ 
        Take list of LLM string outputs (which contain generated QA pairs) and parse into a List of List of tuples.
        Length of output same as length of input.
        """
        all_parsed = []

        for output in output_strings:

            json_str = output.split("=")[0].strip()
            data = json_repair.repair_json(json_str, return_objects=True)
            if not data:
                all_parsed.append([])
                continue

            parsed_pairs = []
            if "q" in data:
                try:
                    parsed_pairs.append((data["q"], data.get("a", "")))
                except: 
                    pass
                    import ipdb; ipdb.set_trace()
            else:
                try:
                    question_keys = sorted(
                        [k for k in data if re.fullmatch(r"q\d+", k)],
                        key=lambda x: int(x[1:])
                    )
                    for qk in question_keys:
                        q_num = qk[1:]
                        question = data[qk]
                        try:
                            answer = data[f"a{q_num}"]
                        except:
                            answer = ""
                            # import ipdb; ipdb.set_trace()
                        parsed_pairs.append((question, answer))
                except: 
                    pass
            all_parsed.append(parsed_pairs)

        return all_parsed

def parse_qa_quality(response: str):
    """
    Parses the response from the question scoring prompt.

    Expected format (before the first '=='):
    [EXPLAIN] <explanation text> [SCORE] [[score_integer]]

    Returns:
        explanation (str): The feedback explanation text.
        score (int): The numeric score.

    Raises:
        ValueError: If the response does not conform to the expected format.
    """
    # Split on first occurrence of two or more equal signs
    split_resp = re.split(r"={2,}", response, maxsplit=1)
    core_response = split_resp[0].strip()

    # Regex pattern to extract explanation and score
    pattern = r"\[EXPLAIN\]\s*(.*?)\s*\[SCORE\]\s*(?:\[\[(\d+)\]\]|\[(\d+)\])$"

    match = re.search(pattern, core_response, re.DOTALL)
    if not match:
        # raise ValueError("Response does not match the expected format.")
        print("Response does not match the expected format.")
        return "", -1

    explanation = match.group(1).strip()
    score_str = match.group(2) or match.group(3)
    score = int(score_str)

    return explanation, score

def parse_cot_annotation(response: str):
    """
    Cleans and parses LLM output that should follow this format:
    
    Rationale: <rationale>
    The answer is: <answer>
    Number of hops: <int>=====
    
    It ignores any extra text before 'Rationale:' and after final '====='.
    """
    # Strip unwanted text
    rationale_start = response.find("Rationale:")
    if rationale_start == -1:
        print(colored("No valid start for the rationale being parsed!", "yellow"))
        return {"rationale": "", "answer": "", "num_hops": ""}  # No valid start
    response = response[rationale_start:]

    last_equals = response.rfind("=")
    if last_equals == -1:
        print(colored("No valid end for the rationale being parsed!", "yellow"))
        return {"rationale": "", "answer": "", "num_hops": ""}  # No valid ending
    response = response[:last_equals + 1]

    # Now apply regex
    pattern = (
        r"^Rationale:\s*(?P<rationale>.*?)"
        r"(?:\n|\\n)The answer is:\s*(?P<answer>.*?)"
        r"(?:\n|\\n)Number of hops:\s*(?P<hops>\d+)={5}$"
    )

    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return {"rationale": "", "answer": "", "num_hops": ""}

    return {
        "rationale": match.group("rationale").strip(),
        "answer": match.group("answer").strip(),
        "num_hops": int(match.group("hops"))
    }

def extract_cot_citations_and_quotes(rationale):
    """
    Extracts (quote, citation_id) pairs from rationale.
    Example: "'X did Y' [Article 2]" -> ("X did Y", "2")
    """
    pattern = r"'([^']+)' \[Article (\d+)\]"
    # return re.findall(pattern, rationale)
    return [(quote, int(aid)) for quote, aid in re.findall(pattern, rationale)]

def check_cot_citation_validity(rationale, context_articles):
    """
    Look for nonexistent citations.
    Returns list of (quote, article_id) that are not actually found in the corresponding article.
    """
    erroenous_citations = []

    # Iterate over tuples of quotes and citations
    for quote, article_id in extract_cot_citations_and_quotes(rationale):

        # If article index invalid
        if article_id < 1 or article_id > len(context_articles):
            erroenous_citations.append((quote, article_id))

        # If quote not in the article
        elif quote not in context_articles[article_id - 1]:
            erroenous_citations.append((quote, article_id))

    return erroenous_citations
