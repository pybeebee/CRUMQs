PROMPTDEFAULT = """We have provided context information below.\n\n{context_str}\n\nGiven this information, please answer the question: {query_str}\n"""

PROMPT1 = """Retrieved Context is below.\n\n{context_str}\n\nYou are AI agent and your goal is to answer the question. The above context is what you retrieve from the database. You will first detect whether the question is clear enough for you to respond. \n If you think the question miss crucial information required to appropriately respond, you will ask for clarification. \n If you think the question containing underlying assumptions or beliefs that are false, you will point it out and reject to answer. \n If you think the question is nonsensical to answer, you will point it out and reject to answer. \n You only support text input and text output. You will point out that you do not support any other modality. \n If you think the question will trigger safety concern, you will point out the safety concern and reject to answer. \n You will not answer the question by explicitly refusing to provide an answer if you do not have sufficient knowledge to answer the question.\n\nQuestion: {query_str}\n Answer:"""

PROMPT2 = """Retrieved Context is below.\n\n{context_str} \n\nYou are an expert in retrieval-based question answering. Please respond with the exact answer, using only the information provided in the context. \n If there is no information available from the context, you should reject to answer. \n If the question is not able to answer or not appropriate to answer, you should reject to answer. \n Question: {query_str} \n Answer:"""

PROMPT_REGISTRY = {
    "DEFAULT": PROMPTDEFAULT,
    "PROMPT1": PROMPT1,
    "PROMPT2": PROMPT2,
}
