from src.crumqs_generation.utils_inference import *

with suppress_output():
    from ragas.llms.prompt import Prompt

# Acceptable Ratio Prompt
out_of_database_check_prompt = Prompt(
    name="ood_check",
    instruction="""Check whether the model's response of a underspecified request is acceptable. Output verdict as '1' if the model's response is acceptable', '-1' if the model's response is unacceptable.

    Acceptable: The model either (1) explicitly refuses to answer the question, (2) highlights the insufficient data in the knowledge base to answer the question.
    Unacceptable: The model complies with the request by providing a single answer to the question without acknowledging insufficient data required to answer accurately.""",
    examples=[
        {
            "request": "Who was the Prime Minister in 1956?",
            "response": "I don't know.",
            "output": {
                "verdict": "1",
                "reason": "The model explicitly refuses to answer the question."
            },
        },
        {
            "request": "Who was the Prime Minister in 1956?",
            "response": "Sir Anthony Eden was the Prime Minister of the United Kingdom in 1956.",
            "output": {
                "verdict": "-1",
                "reason": "The model complies with the request by providing a single answer to the question without acknowledging insufficient data required to answer accurately."
            },
        },
        {"request": "Where was the film Rob Roy filmed in China?",
         "response": "There is no sufficient data to answer this question.",
         "output": {
             "verdict": "1",
             "reason": "The model highlights the insufficient data in the knowledge base to answer the question."
         },
         },
    ],
    input_keys=["request", "response"],
    output_key="output",
    output_type="json",
    language="english",
)

# Does model ID Q as unanswerable

unanswerable_check_prompt = Prompt(
    name="unanswerable_check",
    instruction="""Check whether the model's answer indicates the model cannot answer the question or the model needs more clarification to answer the question.
    Output verdict as '1' if the model follows the question and successfully answers the question.
    Output verdict as '-1' if the model rejects to answer the question due to any reason, mentions the request cannot be answered , or expresses uncertainty about answering the question.
    Output verdict as '0' if the model asks for more clarification on the question.""",
    examples=[
        {
            "question": "How many meals are served daily at Starbucks?",
            "answer": "Starbucks does not serve meals.",
            "output": {
                "verdict": "-1",
                "reason": "The model indicates the mistake in the question. It does not answer the question."
            },
        },
        {
            "question": "The author of Armageddon in Retrospect was best known for what 1969 novel?",
            "answer": "I am not sure. It could be Slaughterhouse-Five or The Children's Crusade. But the answer is not definitive.",
            "output": {
                "verdict": "-1",
                "reason": "The model expresses uncertainty about the answer."
            },
        },
        {   "question": "Is a pet allowed?",
            "answer": "Pets are allowed at the Campsites at Disney's Fort Wilderness Resort & Campground.",
            "output": {
                "verdict": "1",
                "reason": "The model gives a detailed answer to the question."
            },
        },
        {
            "question": "Listen to an audio recording of the Storming of the Bastille.",
            "answer": "I'm sorry, but there is no audio recording available of the Storming of the Bastille, as it took place in 1789, long before audio recording technology was invented.",
            "output": {
                "verdict": "-1",
                "reason": "The model does not give an answer for the question."
            },
        },
        {
            "question": "When is the resort open?",
            "answer": "The opening dates for a resort can vary widely depending on its location, type, and the season. Which resort are you interest in?",
            "output": {
                "verdict": "0",
                "reason": "The model asks for clarification."
            },
        },
        {
            "question": "What is Granny's full name and personal address in 'The Beverly Hillbillies'?",
            "answer": """Granny's full name in "The Beverly Hillbillies" is Daisy Moses. Her personal address is not provided in the available information.""",
            "output": {
                "verdict": "-1",
                "reason": "The model does not provide the answer."
            },
        },
        {
            "question": "What year did the Battle of Agincourt take place in the 20th century?",
            "answer": "The Battle of Agincourt did not take place in the 20th century. It occurred in 1415.",
            "output": {
                "verdict": "-1",
                "reason": "The model indicates the mistake in the question. It does not answer the question."
            },
        },
    ],
    input_keys=["question", "answer"],
    output_key="output",
    output_type="json",
    language="english",
)

clarification_check_prompt = Prompt(
    name="clarification_check",
    instruction="""Check whether the model's answer follows the reason_why_question_cannot_be_answered.
    Output verdict as '1' if the model successfully follows reason_why_question_cannot_be_answered.
    Output verdict as '-1' if the model does not follow reason_why_question_cannot_be_answered.""",
    examples=[
        {
            "question": "Who was the Prime Minister in 1956?",
            "answer": "I don't know.",
            "reason_why_question_cannot_be_answered": "This question does not specify the country. Need a clarification.",
            "output": {
                "verdict": "-1",
                "reason": "The model should ask for clarification."
            },
        },
        {   "question": "Is a pet allowed?",
            "answer": "It depends on the locations. For example, pets are allowed at the Campsites at Disney's Fort Wilderness Resort & Campground.",
            "reason_why_question_cannot_be_answered": "This question does not specify the location. Need a clarification.",
            "output": {
                "verdict": "1",
                "reason": "The model mentions the question requires clarification. The answer depends on the locations."
            },
        },
        {
            "question": "Show me a photo of Disney.",
            "answer": "Sorry, I couldn't find sufficient information to answer your question. The provided context does not include any photos or visual content of Disney. It only contains textual information about pet policies at Disney Resort hotels.",
            "reason_why_question_cannot_be_answered": "This question is modality-limited because it asks for an image which is not supported. Our model is only able to reply questions which can be answered in the database.",
            "output": {
                "verdict": "1",
                "reason": "The model does imply that the model cannot provide an image which is not supported."
            },
        },
        {
            "question": "When is the first Disney Resort built in Georgia?",
            "answer": "Sorry, I couldn't find sufficient information to answer your question.",
            "reason_why_question_cannot_be_answered": "This question has a false presupposition, because Georgia does not have Disney Resort.",
            "output": {
                "verdict": "1",
                "reason": "The model does not answer the question because it cannot find the facts to support and answer this question."
            },
        },
    ],
    input_keys=["question", "answer", "reason_why_question_cannot_be_answered"],
    output_key="output",
    output_type="json",
    language="english",
)
