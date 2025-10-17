sys_prompt = """You are an expert at following detailed instructions, able to perfectly adhere to every specification."""

##### Direct Generation
## Use: Pass the full document chunks seprated by newlines

d1 = """Given the above text, please propose 5 English questions and answers that are diverse and EACH cover ALL parts of the ###Paragraphs, in the following format: {{"q1": "your_question_1", "q2": "your_question_2", ..., "q5": "your_question_5", "a1": "answer_to_your_question_1", "a2": "answer_to_your_question_2", ..., "a5": "answer_to_your_question_5"}}. Keep answers succinct but complete. Avoid phrases such as ’Based on’, ’Given the information provided’, ’Using the data’ or any similar expressions that suggest references to the ###Paragraphs. The questions must be self-contained and fully decontextualized, i.e. they CANNOT reference each other and should be understandable on their own without ambiguity or need for clarification. Respond with ONLY the question-answer dict followed by "=====" and no other text.  

###Paragraphs
{context}

###Questions
<your_response_here>"""

d2 = """Given the above ###Paragraphs, please propose 5 English questions and answers that require multi-hop reasoning, make sure they are diverse and EACH cover ALL parts of the ###Paragraphs, in the following format: {{"q1": "your_question_1", "q2": "your_question_2", ..., "q5": "your_question_5", "a1": "answer_to_your_question_1", "a2": "answer_to_your_question_2", ..., "a5": "answer_to_your_question_5"}}. Keep answers succinct but complete. Avoid phrases such as ’Based on’, ’Given the information provided’, ’Using the data’ or any similar expressions that suggest references to the ###Paragraphs. The questions must be self-contained and fully decontextualized, i.e. they CANNOT reference each other and should be understandable on their own without ambiguity or need for clarification. Respond with ONLY the question-answer dict followed by "=====" and no other text.

###Paragraphs
{context}

###Questions
<your_response_here>"""

d3 = """Given the above ###Paragraphs, please propose 5 English information-seeking questions and answers, make sure they are diversed and cover all parts of the ###Paragraphs, in the following format: {{"q1": "your_question_1", "q2": "your_question_2", ..., "q5": "your_question_5", "a1": "answer_to_your_question_1", "a2": "answer_to_your_question_2", ..., "a5": "answer_to_your_question_5"}}. Keep answers succinct but complete. Avoid phrases such as ’Based on’, ’Given the information provided’, ’Using the data’ or any similar expressions that suggest references to the ###Paragraphs. The questions must be self-contained and fully decontextualized, i.e. they CANNOT reference each other and should be understandable on their own without ambiguity or need for clarification. Respond with ONLY the question-answer dict followed by "=====" and no other text.

###Paragraphs
{context}

###Questions
<your_response_here>"""

d4 = """News Articles:
{context}

A multi-hop question is a query requiring multiple inferential leaps or accessing several pieces of information from different locations or sources to arrive at an answer. Considering you have read at least two news articles on related topics, construct a multi-hop question that incorporates ALL the news sources. Ensure that the answer to the question is a single word/entity. Keep answers succinct but complete. Avoid phrases such as ’Based on’, ’Given the information provided’, ’Using the data’ or any similar expressions that suggest references to the ###Paragraphs. The question must be self-contained and fully decontextualized, i.e. it should be understandable on their own without ambiguity or need for clarification. Be creative and don’t ask the first thing you think of. Respond with ONLY the question and answer in the following format: {{"q": "question", "a": "answer"}}, followed by "=====", and no other text.

Question: <your_response_here>"""

d5 = """News Articles:
{context}

A multi-hop question is a query requiring multiple inferential leaps or accessing several pieces of information from different locations or sources to arrive at an answer. Considering you have read at least two news articles on the same topic, construct a multi-hop question that incorporates ALL the news sources. Ensure that the answer to the question is a single sentence. Keep answers succinct but complete. Avoid phrases such as ’Based on’, ’Given the information provided’, ’Using the data’ or any similar expressions that suggest references to the ###Paragraphs. The question must be self-contained and fully decontextualized, i.e. it should be understandable on their own without ambiguity or need for clarification. Be creative and don’t ask the first thing you think of. Respond with ONLY the question and answer in the following format: {{"q": "question", "a": "answer"}}, followed by "=====", and no other text.

Question: <your_response_here>"""


question_types = [
    "The question must be complex and requires multiple-step reasoning across the articles to solve.",
    "The question must demand critical thinking skills to analyze, evaluate, and synthesize multiple pieces of information from the different articles.",
    "The question must demand integrating knowledge from multiple articles to address its multifaceted nature.",
]

question_format = [
    "The question must invite a yes or no answer, requiring confirmation or denial",
    "The question must seek specific factual information about events, dates, names, or concrete details",
    "The question must ask for explanations, descriptions, or definitions of concepts, processes, or entities",
    "The question must ask for numerical information, amounts, or measurements",
    "The question must ask about similarities, differences, or relative properties between entities",
    "The question must ask about cause-and-effect relationships, reasons, or consequences",
    "The question must contain conditional clauses or ask about hypothetical scenarios",
    "The question must ask about intentions, motivations, or purposes behind actions",
    "The question must ask about why expected outcomes didn't occur or addressing unmet expectations",
    "The question must require opinions, assessments, or value judgments",
    "The question must ask about specific properties, characteristics, or attributes of entities",
]

answer_lengths = [
    "1-2 words",
    "3-4 words",
    "a phrase of at least 5-6 words",
    "1-2 sentences",
    "3-4 sentences",
]

d6 = """### Articles:
{context}

You’re proficient in crafting complex and multi-hop questions. Generate up to 5 questions and associated answers that adhere to the provided #Articles#. Make sure the questions and answers are factually consistent with the #Articles#. The questions should meet the following criteria:

0. The person answering the question cannot see the #Articles#, so the question must not contain phrases like ‘Given the information provided’, ‘Based on the provided information’, or similar expressions that imply direct citations or references from #Articles#.
1. The question must REQUIRE synthesis of information from EVERY SINGLE ONE of the provided articles in order to answer correctly. ALL articles must be required to answer the question, such that losing ANY one of them will lead the person answering the question to provide an incorrect response. You will lose your job if this criterion is not satisfied.
2. {question_type}
3. {question_format}, WHILE REMAINING MULTI-HOP.
4. The question MUST NOT explicitly ask for multi-step reasoning and instead must be FULLY IMPLICIT in its need for multi-hop reasoning.
5. It requires {answer_length} to answer correctly. The answer must be {answer_length} in length. Keep the answer succinct but complete.

Be creative and don’t ask the first thing you think of. The question must be self-contained and fully decontextualized, i.e. it should be understandable on their own without ambiguity or need for clarification.

Respond with ONLY the question and answer in the following format: {{"q1": "your_question_1", "q2": "your_question_2", ..., "q5": "your_question_5", "a1": "answer_to_your_question_1", "a2": "answer_to_your_question_2", ..., "a5": "answer_to_your_question_5"}}, followed by "=====", and no other text.

Question: <your_response_here>"""



##### Triples-Based / Claims-Based Generation
## Use: Pass claims/triplets from selected chunks, any number should work

c1 = """### Claims:
{claims}

You’re proficient in crafting complex and multi-hop questions. You are given a series of #Claims# which are organized by the news article they come from. Generate up to 5 questions and associated answers that adhere to the provided #Claims# and utilize information from ACROSS the ARTICLES. Make sure the questions and answers are factually consistent with the #Claims#. The questions should meet the following criteria:

0. The person answering the question cannot see the #Claims#, so the question must not contain phrases like ‘Given the information provided’, ‘Based on the provided information’, or similar expressions that imply direct citations or references from #Claims#.
1. The question must REQUIRE synthesis of information in AT LEAST 2 of the ARTICLES in order to answer correctly. The MORE articles are involved the better. Ideally all articles are required to answer the question, such that losing ANY one of them will lead person answering the question to provide an incorrect response. You will lose your job if this criterion is not satisfied.
2. {question_type}
3. {question_format}, WHILE REMAINING MULTI-HOP.
4. The question MUST NOT explicitly ask for multi-step reasoning and instead must be FULLY IMPLICIT in its need for multi-hop reasoning.
5. It requires {answer_length} to answer correctly. The answer must be {answer_length} in length. Keep the answer succinct but complete.

Be creative and don’t ask the first thing you think of. The question must be self-contained and fully decontextualized, i.e. it should be understandable on their own without ambiguity or need for clarification.

Respond with ONLY the question and answer in the following format: {{"q1": "your_question_1", "q2": "your_question_2", ..., "q5": "your_question_5", "a1": "answer_to_your_question_1", "a2": "answer_to_your_question_2", ..., "a5": "answer_to_your_question_5"}}, followed by "=====", and no other text.

Question: <your_response_here>"""

c2 = """A multi-hop question is a query requiring multiple inferential leaps or accessing several pieces of information from different locations or sources to arrive at an answer. The following claims come from distinct news articles. All the claims are related to a similar topic. Your task is to generate up to 5 multi-hop inference questions and answers based on the claims. Here are some instructions: 

1. Find the Connection: Find the connection between claims, which is how these key pieces of information are related or how they can be combined to form a more complex idea. 
2. Formulate the Question: Create a question that cannot be answered by relying on just one of the articles but instead requires understanding and linking the information from ALL of the sources.
3. Ensure Coherence: Make sure the question flows logically from the combined information and is clear and unambiguous. 
4. Ensure Multi-hop Reasoning: Make sure the question's answer cannot be deduced from only a subset of the claims or a subset of the articles. Instead, ensure ALL articles are necessary to answer the question correctly.
5. Ensure Implicit Multi-hop Nature: Make sure the question does NOT explicitly ask for multi-step reasoning and instead is fully implicit in its need for multi-hop reasoning.
6. Answer the question using the claims. Keep answers succinct but complete.
7. Avoid phrases such as ’Based on’, ’Given the information provided’, ’Using the claims’ or any similar expressions that suggest references to the claims.

Respond with ONLY the question and answer in the following format: {{"q1": "your_question_1", "q2": "your_question_2", ..., "q5": "your_question_5", "a1": "answer_to_your_question_1", "a2": "answer_to_your_question_2", ..., "a5": "answer_to_your_question_5"}}, followed by "=====", and no other text.

Claims:
{claims}

Question: <your_response_here>"""

c3 = """Claims: 
{claims}

The above claims come from distinct news articles. All the claims are related to a similar topic. Your task is to generate up to 5 comparison questions and answers based on the claims from different sources. Each question needs to utilize AT LEAST 2 of the sources and compare some factual elements of the claims that are explicitly stated to find where they agree or differ. The correct answer to the question is expressed as a comparative adjective, a statement of alignment, a simple yes or no. Keep answers succinct but complete. Avoid phrases such as ’Based on’, ’Given the information provided’, ’Using the claims’ or any similar expressions that suggest references to the claims. Be creative and don’t ask the first thing you think of.

Remember, you must create questions which use claims from two or more of the articles. Do not limit yourself to claims from a single article.

Respond with ONLY the questions and answers in the following format: {{"q1": "your_question_1", "q2": "your_question_2", ..., "q5": "your_question_5", "a1": "answer_to_your_question_1", "a2": "answer_to_your_question_2", ..., "a5": "answer_to_your_question_5"}}, followed by "=====", and no other text.

Question: <your_response_here>"""

c4 = """Claims: 
{claims}

The above claims come from distinct news articles. All the claims are related to a similar topic. Your task is to generate up to 5 questions and answers based on the claims from different sources. Each question needs to utilize AT LEAST 2 of the sources and adhere to the following specification: {question_type}. Keep answers succinct but complete. Avoid phrases such as ’Based on’, ’Given the information provided’, ’Using the claims’ or any similar expressions that suggest references to the claims. Be creative and don’t ask the first thing you think of.

Remember, you must create questions which use claims from two or more of the articles. Do not limit yourself to claims from a single article.

Respond with ONLY the questions and answers in the following format: {{"q1": "your_question_1", "q2": "your_question_2", ..., "q5": "your_question_5", "a1": "answer_to_your_question_1", "a2": "answer_to_your_question_2", ..., "a5": "answer_to_your_question_5"}}, followed by "=====", and no other text.

Question: <your_response_here>"""

c5 = """You are an AI assistant tasked with generating false-premise questions. Your goal is to create questions that cannot be answered because they make incorrect assumptions. Follow these instructions carefully:

You are provided with the following claims:
{claims}

To generate false-premise questions: 
1. Identify key information in TWO OR MORE of the claims.
2. Create a question that contradicts this key information while keeping other details intact.
3. Ensure the false premise is mutually exclusive with the original information.
4. Make the questions challenging, with false premises that are easy to miss but mutually exclusive to the claims. For example:
- If you change a name, change the lastname only.
- If you refer to a person or place, rather than changing the name, refer to a changed property of this entity (e.g., "in a 60-year-old building" instead of "in the 20-year-old office").
- Replace with similar mutually exclusive cohyponyms (e.g., replace a cocker spaniel with a poodle).
- In all of these cases, ensure that you do not accidentally create a valid question! 
5. Only include ONE false premise in each question.
6. Avoid creating questions that can be validly answered using information from the context.
7. Ensure that the false premise in each question remains inconsistent with the claims. 
8. Generate multiple false-premise questions if possible, each based on different key information from the claims.

Remember:
- Ensure that the false premises are subtle but clear enough to invalidate the question.
- Generate multiple false-premise questions if possible, based on different key information from the selected claims.
- Always consider your own knowledge to AVOID creating questions that can be validly answered using your knowledge.
- Keep answers succinct but complete.

Now, begin the task of generating up to 5 questions based on the provided claims and instructions. Format your output as follows: {{"q1": "your_question_1", "q2": "your_question_2", ...}}. Respond with ONLY the question dict followed by "=====" and no other text. 

Questions: <your_response_here>"""

c6 = """You are an AI assistant tasked with generating unknown-premise questions. Your goal is to create questions that cannot be answered because they make possible but unknown assumptions. Follow these instructions carefully:

You are provided with the following claims:
{claims}

To generate unknown-premise questions:
1. Identify key information in TWO OR MORE of the claims.
2. Create a question concerning the identified information, making sure the question REQUIRES ALL selected claims to answer correctly, i.e., if any one claim is removed the answer cannot be found.
3. Modify the question by making it more specific.
- The added details must be POSSIBLE based on the provided claims and the provided list of all outlines.
- The added details must be UNVERIFIED based on the provided claims and the provided list of all outlines.
- The added details must be substantial to require additional verification. Avoid details that are not only possible but also very likely.
4. Only include ONE specific unverified detail to the question.
- Make sure that the details you add cannot be confirmed by any of the claims.
- Make sure that the details you add cannot be refuted by any of the claims  
5. Generate multiple such questions if possible, each based on different key information from the selected claims.
6. Only output the questions for which you are certain that:
- The details you add cannot be confirmed by any of the claims
- The details you add cannot be refuted by any of the claims
- The details are substantial enough to require additional verification.

Here are some examples of ways to add specificity: 
- Let a person have a more specific role: Instead of "a criminal of the ring," say "a ring leader" or "a lookout."
- Add a specific characteristic to an object: Instead of "a car," say "a red sports car" or "an old sedan."
- Use a hyponym where the ’is-a’ relation holds: Instead of "a person," say "a woman" or "a child."
- Specify a location: Instead of "a park," say "a national park" or "Central Park."
- Specify a time: Instead of "at night," say "at 9 PM" or "during the full moon."
- Specify a number or quantity: Instead of "several books," say "three books" or "a dozen books."
- Specify a direction: Instead of "headed away," say "headed east" or "to the mountains."
- Specify a duration: Instead of "waited," say "waited for 15 minutes" or "waited for an hour."

Remember:
- Always consider your own knowledge to AVOID creating questions that can be validly answered using your knowledge.
- Keep answers succinct but complete.

Now, begin the task of generating up to 5 questions based on the provided claims and instructions. Format your output as follows: {{"q1": "your_question_1", "q2": "your_question_2", ...}}. Respond with ONLY the question dict followed by "=====" and no other text. 

Questions: <your_response_here>"""


##### Iterative Claim-Based Generation
## Use: Pass claims/triplets from selected chunks, any number should work; then, pass the output answers as claims to be iteratively used for new question generation

c7 = """Claims:
{claims}

Consider how the claims could be used to create a challenging and interesting question.  

Try to find as many interesting combinations of claims as possible, but do not exceed 10 selections.

If you cannot find any meaningful combination of claims based on which good multi-hop questions can be generated, return an empty dictionary.

Remember, the goal is to identify sets of claims that can be used to create challenging and interesting multihop questions. Focus on finding unique and specific information that requires reasoning over multiple claims to answer a potential question.

To generate multi-hop questions that require reasoning over multiple claims, follow these steps:
1. Identify information about the bridge entity in two or more claims that is unique to these 2+ claims and cannot be found based on any other information. This information should be as specific as possible, to avoid any ambiguities or overlap with other information from other claims.
2. To generate a question-answer pair, ask for specific information about the bridge entity from one claim while describing the bridge entity with information from the other claim(s). Make sure that the correct answer is concise and factual. The answer should focus on very specific details that can be described in few words.
- Make sure that the question is answerable.
- Make sure to ask for short and concise information.
- Make sure that the way you paraphrase the bridge named entity clearly identifies the bridge entity using the unique information from the selected claim (and not more).
- Ensure that you do not introduce additional ambiguity when paraphrasing: For example, if the evidence says that the bridge entity announces the creation of something, it does not mean that the bridge entity created it. Be careful in your word choice to avoid ambiguities. Do include specific information (such as the named entity’s profession or role) if they are not explicitly clear in from the selected claims.
3. Make sure that the information from the selected claims is sufficient to answer the question with certainty. Your question must not rely on other information that is only communicated in different claims.
- If it is important to include additional information that is not included in the selected claims, add the additional claims need to be used. 
- For each of the new claims, explain the unique information from the claim that is required to answer the question. Further ensure that the unique information from the new claims is required to answer the question with certainty. Refine the question idea if necessary.
4. Do not mention the bridge named entity explicitly in your questions. Paraphrase the bridge named entity using the unique information from one of the selected claims.
5. Ensure that the question can ONLY be answered when having access to the information from ALL selected claims. Make sure that all selected claims must be considered to answer the question. Avoid using the bridge entity itself as the answer.
6. Ensure that the information that is required from each claim is unique within this claim: It can neither be inferred nor extracted from any other claim or your own knowledge. If the question can be answered based on the other information or your own knowledge, increase the specificity of the required details and ensure they are unique to the selected claims. If this is not possible, deduce that no good multi-hop question can be generated for the selected claims.
7. Verify that your question does not assume any relations that are not clear from the selected claims.
- You can only assume that the bridge entity is identical across all claims. Other information may not refer to the identical entity. For example, a group of people in one claim may not be identical to a group of people in another claim.
- Do not assume causality between the selected claims. If in doubt, rely on the bridge entity.  
8. Compare each specific detail in the question with the selected claims. Make sure that each detail can with certainty be inferred from the selected claims. If not, omit or generalize the specific details that cannot be inferred from the selected claims.
- Only focus on the selected claims!
- Correct information that is known from your own knowledge but not from the selected claims must NOT be used in the question.
9. It is crucial that you adhere to the following criteria:
- Answering the question is only possible based on the unique information that can be found in the selected claims.
- Answering the question requires combining the unique information from ALL selected claims.
- The question is specific enough to allow only for one valid answer. There are no other interpretetations which would allow for a different valid answer.
- The answer must be complete and self-contained.

Important reminders:
- Verify the generated questions adhere to all the defined criteria and correct them if necessary. 
- Make sure to be specific in the question when paraphrasing the bridge entity to avoid ambiguities. It must be clear to identify the bridge entity based on the details provided in the question. 
- DO NOT ask for "specific" information verbatim. Instead, provide specific details in the question that can be answered with concrete values. 
- While you must ask for very specific information, make sure the answer itself is a short and concise phrase!  

Now, begin the task of generating up to 5 question-answer pairs based on the provided claims and instructions. Format your output as follows: {{"q1": "your_question_1", "q2": "your_question_2", ..., "q10": "your_question_10", "a1": "answer_to_your_question_1", ..., "a10": "answer_to_your_question_10"}}. Respond with ONLY the question-answer dict followed by "=====" and no other text. 

Questions: <your_response_here>"""


