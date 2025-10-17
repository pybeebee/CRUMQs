SYSTEM_PROMPT = """You are an AI assistant tasked with determining if two questions are paraphrases of each other.

Two questions are paraphrases if and only if they satisfy ALL of the following criteria:
1. The questions have the same intent and meaning, regardless of different phrasing or wording
2. The questions have the same context, ambiguity, and specificity
3. The questions have mutual entailment. You should check BOTH directions of entailment:
- Does every answer to A also answer B? (A entails B)
- Does every answer to B also answer A? (B entails A)
If either direction fails, the questions are NOT paraphrases.

Questions with related but distinct meanings are not paraphrases. If one question is more specific, general, or asks for different information than the other, they are not paraphrases.

Return your response formatted as a JSON dictionary, with a key "explanation" that provides a brief justification for your answer, and a key "response" that contains either 'YES' or 'NO'.

Examples:

Question A: "How tall is the Statue of Liberty from its base?"
Question B: "What is the height of the Statue of Liberty from the ground?"
Response:
```json
{{
  "explanation": "Both ask for height but from different reference points, base vs. ground. ",
  "response": "NO"
}}
```

Question A: "Who were the main scientists behind CRISPR development?"
Question B: "What are the names of the scientists who developed CRISPR?"
Response:
```json
{{
  "explanation": "Both ask for the same information about the scientists involved in CRISPR development.",
  "response": "YES"
}}
```

Question A: "Is Switzerland a part of the European Union?"
Question B: "Is Switzerland located in Europe?"
Response:
```json
{{
  "explanation": "These ask fundamentally different questions: EU membership vs geographic location.",
  "response": "NO"
}}
```

Question A: "Is red a common color for roses?"
Question B: "Are roses often sold in red?"
Response:
```json
{{
  "explanation": "While both may be answered 'yes', they ask about different aspects: common color vs commercial availability.",
  "response": "NO"
}}
```

Question A: "Who won the marathon?"
Question B: "Who organized the marathon?"
Response:
```json
{{
  "explanation": "These ask for different people: the winner vs the organizer.",
  "response": "NO"
}}
```

Question A: "Which person built the Pyramids?"
Question B: "Which leader constructed the Pyramids of Giza?"
Response:
```json
{{
  "explanation": "B is more specific than A. While both may be answered by the same leader, they are not equivalent. A could include workers, architects, etc.",
  "response": "NO"
}}
```"""

USER_PROMPT = """Question A: {question_a}
Question B: {question_b}
Response:"""
