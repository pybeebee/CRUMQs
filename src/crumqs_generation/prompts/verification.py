##### Entailment between evidences and claims
claim_verif_factcg = """{document}

Choose your answer: based on the paragraph above can we conclude that "{claim}"?  

OPTIONS: 
- YES 
- NO 

Respond ONLY with YES or NO. Do not explain or provide any other text."""


triple_verif_factcg = """{document}

Choose your answer: based on the paragraph above can we conclude the relation {triple}?  

OPTIONS: 
- YES 
- NO 

Respond ONLY with YES or NO. Do not explain or provide any other text."""


##### Annotate each question with reasoning steps to answer it
## Use: Format context as Document [1]: ...\n\nDocument [2]: ...

cot_rationale = """### Articles
{context}

### Question
{question}

### Answer
{answer}

Write an accurate and concise rationale for the given question using only the provided articles (some of which might be irrelevant) and cite them properly. Start with an accurate, engaging, and concise explanation based only on the provided articles. Must end with "The answer is:". Use an unbiased and journalistic tone. Always cite and extract word-for-word quotes for any factual claim, and place the citation immediately after the quote (e.g., "fact from article 2" [Article 2]). Do not paraphrase or invent quotes or citations. If no quote can be found to support a claim, do not include that claim. Do not cite an article unless it contains a directly quoted statement used in your rationale. Think step-by-step.

After writing the rationale, estimate how many hops of reasoning are required to answer the question. A hop is defined as a discrete step that requires retrieving and integrating a new piece of information, especially when that information is found in a different article, paragraph, or sentence. Count a hop when the rationale transitions between distinct supporting facts that contribute meaningfully to answering the question.

Format your response as follows, concluding your entire response with "=====":
Rationale: <rationale here>
The answer is: <answer here>
Number of hops: <integer here>=====

Strictly follow the forma and do not output any other text."""


##### Question/Answer Quality Checking

criteria = {
    "Context Necessity": """Context Necessity - Is EACH ARTICLE in the provided context NECESSARY to answer the question? This evaluates whether the question truly demands multi-hop reasoning and fully utilizes the context. Ideal questions should be 100% context-dependent, meaning they CANNOT be accurately answered if ANY ONE of the articles is removed: they should rely on a reasoning chain that spans multiple documents or passages, such that omitting any one would break the logical flow or result in an incomplete answer. In other words, the article set is MINIMALLY necessary to answer the question. A strong multi-hop question should "force" the use of all documents—none should feel skippable or irrelevant. Generic or overly broad questions that could be answered independently of the text (without specialized knowledge) are discouraged.""",
    "Context Sufficiency": """Context Sufficiency - Can the answer to the question be found ENTIRELY within the provided context? This criterion determines whether the context articles contain enough information to answer the question. A question should NOT require external assumptions or knowledge beyond the context articles unless that knowledge is very general or trivial. Questions with answers that are clearly present and verifiable in the text should receive high marks.""",
    "Answer Correctness": """Answer Correctness - Based on the provided documents, is the answer factually correct and appropriate for the given question? This criterion evaluates whether the answer is factually correct and provides an appropriate level of detail to fully satisfy the question. The answer should be reasonable and grounded in the provided context. It does not require exhaustive mention of every possible entity or fact, but it should not neglect details necessary to answer the question correctly.""",
    "Answer Uniqueness": """Answer Uniqueness - Is this the only plausible answer to the given question that can be obtained based on the context? This checks whether the answer is uniquely determined by the information in the context. Ideal answers should be both correct and exclusive given the text. If there are multiple plausible answers or ambiguity remains, a lower score should be issued.""",
} 

rubrics = {
    "Context Necessity": """0: The question can be correctly and fully answered without the context.\n1: The question can be answered without reading the context, but the answer will be incomplete.\n2: The question can be fully answered using a subset of the provided context. Some articles are unnecessary to obtain the answer, redundant, or ambiguous. The question is only partially multi-hop.\n3: Every article is necessary and critical to answer the question correctly. The question requires integrating information across all sources; omitting any single article would result in an incomplete or incorrect answer. It is impossible to answer the question without full knowledge of the context.""",
    "Context Sufficiency": """0: The context does not contain enough information to derive the answer, even when combining multiple parts. Key facts or reasoning steps are missing.\n1: The context contains some necessary pieces to derive the answer, but they are either incomplete, ambiguous, or require additional knowledge.\n2: The context is sufficient to answer the question. The context contains all the necessary information to fully and unambiguously derive the answer. All details required for intermediate steps and facts are present, and no outside knowledge is required.""",
    "Answer Correctness": """0: The model answered the question incorrectly.\n1: The model answered the question but made some inaccuracies.\n2: The model’s answer is completely correct and requires no additions or corrections.""",
    "Answer Uniqueness": """0: Without reading the text, it is possible to give a completely different answer to the given question.\n1: Without reading the text, one can give a different answer that only slightly differs.\n2: Without reading the text, the model’s answer is the only possible one for the given question.
""",
}


question_scoring_template = """###Task Description:
You are a perfect evaluator. You are given a context, a question to evaluate, the gold answer to the question, and a score rubric representing an evaluation criterion.
1. Write a detailed but SUCCINCT feedback that assesses the quality of the question strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score. You should refer to the score rubric. The score must be as described in the rubric. 
3. The output format should look as follows:
[EXPLAIN] <write a succinct feedback and an explanation about whether the question meets the criterion, reasoning step by step and explaining each argument> [SCORE] [[an integer number within the boundaries of the rubrics]]
4. Conclude your entire response with "=====" and DO NOT generate any other text afterward. For example, your final response should appear in the form: 
[EXPLAIN] <your_explanation_here> [SCORE] [[2]]=====

### Context:
{context}

### Question to evaluate:
{question}

### Gold answer:
{answer}

### Criterion name:
{criterion}

### Score rubric:
{rubric}

### Feedback:"""


answer_scoring_template = """###Task Description:
You are a perfect evaluator. You are given a context, a question, an answer to evaluate, and a score rubric representing an evaluation criterion.
1. Write a detailed feedback that assesses the quality of the answer strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score. You should refer to the score rubric. The score must be as described in the rubric. 
3. The output format should look as follows:
[EXPLAIN] <write a feedback and an explanation about whether the answer meets the criterion, reasoning step by step and explaining each argument> [SCORE] [[an integer number within the boundaries of the rubrics]]
4. Conclude your entire response with "=====" and DO NOT generate any other text afterward. For example, your final response should appear in the form: 
[EXPLAIN] <your_explanation_here> [SCORE] [[2]]=====

### Context:
{context}

### Queston:
{question}

### Answer to evaluate:
{answer}

### Criterion name:
{criterion}

### Score rubric:
{rubric}

### Feedback:"""


get_template = {
    "Context Necessity": question_scoring_template,
    "Context Sufficiency": question_scoring_template,
    "Answer Correctness": answer_scoring_template,
    "Answer Uniqueness": answer_scoring_template,
}
