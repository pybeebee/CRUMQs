from ragas.llms.prompt import Prompt

sys_prompt = """You are an expert at following detailed instructions, able to perfectly adhere to every specification."""

topic_extraction_prompt = Prompt(
    name="topic_extraction",
    instruction="""Consider the user story and extract ALL key topics in the following report request, focusing on the most significant aspects. Each extracted topic must be a single phrase. Respond using a json with a single key "topics" whose value is a comma-separated list of topics, with NO other text.\n\n1. Avoid vague, generic topics. **All extracted topics must refer explicitly and precisely to the entities, subjects, or events described in the report request.** For example, for a report request referencing ovarian cancer treatments, do **not** write "medical treatments" but rather "medical treatments for ovarian cancer". Replace pronoun references with the original entity. If any topics include abstract or general words (e.g., "significance", "information", "methods", "impact"), be sure to specify what these words refer to and any key entities (e.g., "commitment to [entity] by [entity]", "impact of [entity] on [entity]" instead of just "commitment to [entity]" or "impact on [entity]").\n2. Each topic must be understandable **without needing access to the report request**.\n3. Identify and the primary topic or category or provide a short description of the main subject matter of the report request.\n4. If there are subtopics or secondary themes mentioned in the report request, identify these as well. If the report request discusses multiple topics, be sure to identify all of these topics.\n5. Consider the context and tone of the report request and the user story to determine the most appropriate topics. Take into account keywords, phrases, or specific terms that relate to the topics. If any notable entities (people, places, brands, products, etc.) are mentioned in the report request, formulate additional topics that clearly include those entities. Similarly, if the report request suggests any actions, decisions, or recommendations related to the topics, create additional topics describing these.\n6. Make sure your extraction is clear, concise, and comprehensive. Do not respond with your own recommendations or feedback. RESPOND WITH THE TOPICS ONLY.""",
    examples=[
        {
            "user_story": "I am a graduate student in environmental studies preparing a seminar presentation on the history of ecological thought. I want to highlight the pivotal role that Rachel Carson played in shifting public perception about chemical pesticides and in laying the groundwork for modern environmental policy in the United States.",
            "report_request": "I'm preparing a report on Rachel Carson’s influence on the environmental movement. Please focus on how her work, especially Silent Spring, contributed to public awareness about pesticide use, and how it shaped U.S. environmental policy in the decades that followed.",
            "output": {
                "topics": [
                    "Rachel Carson's influence on the environmental movement",
                    "impact of Silent Spring on public awareness of pesticides", 
                    "contribution of Rachel Carson to U.S. environmental policy", 
                    "legacy of Rachel Carson's environmental advocacy", "pesticide concerns raised in *Silent Spring", 
                    "public response to Rachel Carson’s environmental work"
                ]
            },
        },
        {
            "user_story": "As a freelance science journalist writing for a bioethics newsletter, I’m covering recent developments in gene-editing technologies. My readers are especially interested in how tools like CRISPR are being applied in clinical settings and the ethical dilemmas these innovations pose for medicine and regulation.",
            "report_request": "Please create a report on the recent advancements in CRISPR gene-editing technology, especially focusing on its application to human genetic diseases. I’m also interested in the ethical debates and regulatory responses that have emerged around its clinical use.",
            "output": {
                "topics":[ 
                    "recent advancements in CRISPR gene-editing technology",
                    "applications of CRISPR to human genetic diseases",
                    "ethical debates about clinical use of CRISPR", 
                    "regulatory responses to CRISPR gene-editing", 
                    "impact of CRISPR technology on genetic medicine",
                    "controversies surrounding human genome editing"
                ]
            },
        },
        {
            "user_story": "I’m an art history instructor developing teaching materials for a university course on modern visual culture. I need a concise but comprehensive overview of how street art in European cities evolved as a medium of political commentary, including its cultural significance, public reception, and leading figures over the past two decades.",
            "report_request": "Summarize the evolution of street art in urban Europe, especially focusing on the role of political messaging, public reaction, and notable artists who helped shape its cultural impact in the 2000s and 2010s.",
            "output": {
                "topics": [
                    "evolution of street art in urban Europe", 
                    "political messaging in European street art",
                    "public reaction to street art in European cities", 
                    "cultural impact of European street art in the 2000s and 2010s", 
                    "notable street artists in Europe", 
                    "notable street artists in Europe in the 2000s and 2010s", 
                    "role of street art in urban political expression"
                ]
            },
        },
    ],
    input_keys=["user_story", "report_request"],
    output_key="output",
    output_type="json",
)

# Prompt for LiteLLM
doc_grounded_topic_extraction_prompt = Prompt(
    name="doc_grounded_topic_extraction",
    instruction="""Consider the theme. Extract the top 1-5 topics from the provided TEXT, focusing on the aspects that are RELEVANT to the theme. Respond using a json with a single key "topics" whose value is a comma-separated LIST of topics, with NO other text. DO NOT include the '[' or ']' symbol. DO NOT include topics NOT relevant to the theme. Avoid vague, generic topics. **All extracted topics must refer explicitly and precisely to the entities, subjects, or events described in the TEXT and be related to the theme.** For example, for a text referencing stomach cancer treatments, do **not** write "medical treatments" but rather "medical treatments for stomach cancer". Replace pronoun references with the original entity. If any topics include abstract or general words (e.g., "significance", "information", "methods", "impact"), be sure to specify what these words refer to and any key entities (e.g., "commitment to [entity] by [entity]", "impact of [entity] on [entity]" instead of just "commitment to [entity]" or "impact on [entity]"). Each topic must be understandable **without needing access to the report request**. MAKE SURE ALL TOPICS ARE CLEARLY RELEVANT TO THE THEME. Each extracted topic must be a single phrase. Make sure your extraction is clear, concise, and comprehensive and from the TEXT. Do not respond with your own recommendations or feedback.""",
    examples=[
        {
            "theme": "Rachel Carlson influence pesticides, policy",
            "text": "While many historians trace the roots of environmental activism to the conservationist movements of the early 20th century, a major turning point came with the publication of Rachel Carson’s *Silent Spring* in 1962. Though Carson had a background in marine biology, her writing drew widespread attention to the dangers of synthetic pesticides. Around the same time, concerns about air pollution and nuclear fallout were also gaining public visibility, contributing to a broader wave of environmental concern.",
            "output": {
                "topics": [
                    "Rachel Carson and synthetic pesticides",
                    "public awareness of pesticide risks after Rachel Carlson's *Silent Spring*",
                    "Rachel Carson’s contribution to environmental activism through *Silent Spring*"
                ]
            },
        },
        {
            "theme": "CRISPR applications and ethics, regulation",
            "text": "CRISPR has revolutionized research in everything from agriculture to virology. Scientists are using it to engineer virus-resistant crops and to develop new diagnostics for infectious diseases. However, its application in human medicine — especially for rare genetic disorders — has raised ethical concerns. Some critics argue that regulatory bodies have been too slow to address these challenges, while others call for global coordination.",
            "output": {
                "topics":[ 
                    "CRISPR's revolutionary effects",
                    "CRISPR and agriculture",
                    "CRISPR disease diagnosis",
                    "ethical concerns around clinical applications of CRISPR",
                    "regulatory challenges in human gene-editing"
                ]
            },
        },
        {
            "theme": "European street art politics, reactions, artists",
            "text": "Street art has taken many forms across the globe — from stencil graffiti in Buenos Aires to large-scale murals in Johannesburg. In Europe, artists like Banksy helped propel the genre into the mainstream, though critics often debate whether his work should be considered commercial art. Meanwhile, some city governments have launched initiatives to support street artists through legal wall spaces, though others still impose fines for unsanctioned pieces.",
            "output": {
                "topics": [
                    "political messaging in European street art",
                    "Banksy’s role in mainstreaming European street art",
                    "public and governmental responses to street art in Europe"
                ]
            },
        },
    ],
    input_keys=["theme", "text",],
    output_key="output",
    output_type="json",
)


doc_context_relevance = Prompt(
    name="doc_context_relevance",
    instruction="""You are given a topic, a context, a user story, and a report request. Is any part of the context related or somewhat related to the main focus of the topic, report request, and/or user story? For example, a topic about Machu Picchu and a context about language models or a context with only paper citations is TOTALLY irrelevant; a topic about Machu Picchu construction and a context about Vega family residence construction, about software architecture, or about Choquequirao are IRRELEVANT; a topic about analyzing the Munchinson Meteorite COULD be relevant to a context about analyzing the Ischgl meteorite; however, a topic about discovering the Murchinson Meteorite and a context generally about the Ischgl meteroite are NOT RELEVANT. Respond YES or NO and no other text.""", 
    input_keys=["topic", "user_story", "report_request", "context"],
    output_key="output",
    output_type="str",
)

answerable_question_prompt_all = Prompt(
    name="seed_question",
    instruction="""You are given a topic, a context, a user story, and a report request.\nGenerate up to 10 specific, non-generic questions about DETAILS in the context that are related to the *TOPIC*. Respond using a json with a single key "questions" whose value is a comma-separated STRING list of questions, with NO other text. Each question MUST have punctuation (e.g., ?) at the end. Enclose the entire string value of the key "questions" in quotation marks. If the content of a question uses a quotation mark, use the single quote ' instead; within individual questions, do not include double quotes like ".\nIMPORTANTLY: (1) Make sure the questions can be fully answered from given context. (2) Make sure the question is self-contained and can be understood without the context, i.e. any key entities needed to understand the question are present. (3) MAKE SURE the questions are NOT too high-level and are about the TOPIC. (4) MAKE SURE each question addresses some part of the report request in a way that aligns with the user story.\nBe sure to satisfy all criteria above. DO NOT include references such as "the context", "the user story", or "the report request" in the question. Avoid vague or surface-level questions. Diversify the length and style of questions. SIMPLE SHORT QUESTIONS ARE GOOD TOO. Include both short and longer questions.""", 
    input_keys=["context", "topic", "user_story", "report_request"],
    output_key="output",
    output_type="json",
)


seed_external_answerability_prompt = Prompt(
    name="seed_external_answerability",
    instruction="""Provide a correct or plausible answer to the question using the information from the given context. Your answer should be self-contained; DO NOT include references such as "the context". If no question is given, answer the empty string. Output verdict as '1' if an answer is able to be formulated from the context and '-1' otherwise or if no question is given. Respond using a json with a two keys "answer" and "verdict" whose values are STRINGS, with NO other text.""",
    examples=[
        {
            "context": """Climate change is significantly influenced by human activities, notably the emission of greenhouse gases from burning fossil fuels. The increased greenhouse gas concentration in the atmosphere traps more heat, leading to global warming and changes in weather patterns.""",
            "question": "How do human activities contribute to climate change?",
            "answer": {
                "answer": "Human activities contribute to climate change primarily through the emission of greenhouse gases from burning fossil fuels. These emissions increase the concentration of greenhouse gases in the atmosphere, which traps more heat and leads to global warming and altered weather patterns.",
                "verdict": "1",
            },
        },
        {
            "context": """The concept of artificial intelligence (AI) has evolved over time, but it fundamentally refers to machines designed to mimic human cognitive functions. AI can learn, reason, perceive, and, in some instances, react like humans, making it pivotal in fields ranging from healthcare to autonomous vehicles.""",
            "question": "What are the key capabilities of artificial intelligence?",
            "answer": {
                "answer": "Artificial intelligence is designed to mimic human cognitive functions, with key capabilities including learning, reasoning, perception, and reacting to the environment in a manner similar to humans. These capabilities make AI pivotal in various fields, including healthcare and autonomous driving.",
                "verdict": "1",
            },
        },
        {
            "context": """The novel "Pride and Prejudice" by Jane Austen revolves around the character Elizabeth Bennet and her family. The story is set in the 19th century in rural England and deals with issues of marriage, morality, and misconceptions.""",
            "question": "What year was 'Pride and Prejudice' published?",
            "answer": {
                "answer": "The answer to given question is not present in context",
                "verdict": "-1",
            },
        },
        {
            "context": """When the gold-hungry Spanish invaders reached Cusco, they were no doubt delighted to lay eyes on Qorikancha, a complex that included the Empire\u2019s more important temple to the sun god Inti. The site also hosts the annual Inti Raymi festival, a reenactment of the Inca\u2019s winter solstice celebration. 
            
            Nestled between towering mountain peaks much like Machu Picchu, the Qorikancha complex features the Inca\u2019s reinforced terraces and a mix of both finished and unfinished structures, providing a unique look at the processes undertaken to build such sites. """,
            "question": "What role did the Inti Raymi festival play in the design and construction of Machu Picchu?",
            "answer": {
                "answer": "",
                "verdict": "1",
            }
        }
    ],
    input_keys=["context", "question"],
    output_key="answer",
    output_type="json",
    language="english",
)


unanswerable_check_prompt = Prompt(
    name="unanswerable_check",
    instruction="""You will compare a given answer with a given ground truth in response to the question, adhering to the following guidelines:\n\nIf the answer includes all the core ideas from the ground truth and adds only relevant or consistent information, output verdict as "1". Do not require exact wording, structure, or content — the answer may be rephrased, more detailed, present MORE information, or differently structured, as long as it appropriately answers the question in a similar fashion to the ground truth.\n\nIf the answer is off-focus from the *question*, omits key points or details in the ground truth (see caveat below), or clearly introduces contradictions, output verdict as "-1". If the answer suggests a lack of knowledge or information, output verdict as "-1". If the **question** is underspecified, leading to multiple possible answers, output verdict as "-1".\n\nIf the **ground truth** indicates a lack of knowledge while the **answer** appropriately responds to the question (i.e., does not indicate lack of knowledge), output verdict as "0". If the **question** is NOT underspecified yet both the ground truth and answer are plausible responses to the question, EVEN IF the answer does not capture all key points of the ground truth, output verdict as "0". If the **ground truth** is the empty string "" or blank while the **answer** appropriately responds to the question (i.e., does not indicate lack of knowledge), output verdict as "0".\n\nRespond using a JSON with two keys: "reason" and "verdict", whose values are STRINGS. Output nothing else.""",
    examples=[
        {
            "question": "What human capital management initiatives has Disney reported on in 2020?",
            "answer": """I don't know. Not enough information.""",
            "ground_truth": "Diversity, Equity, and Inclusion (DE&I) initiatives, health, wellness, family resources.",
            "output": {
                "reason": "The answer indicates a lack of knowledge.",
                "verdict": "-1",
            },
        },
        {
            "question": "What human capital management initiatives has Disney reported on in 2020?",
            "answer": """The information provided does not specifically address Disney's capital management initiatives.""",
            "ground_truth": "Diversity, Equity, and Inclusion (DE&I) initiatives, health, wellness, family resources.",
            "output": {
                "reason": "The answer indicates a lack of information to respond to the question.",
                "verdict": "-1",
            },
        },
        {
            "question": "What fiscal year is covered in the 2023 SEC report on Disney?",
            "answer": """The United States Securities and Exchange Commission report provides information about Disney's fiscal year ending on October 2, 2021.""",
            "ground_truth": "The United States Securities and Exchange Commission report provides information about Disney's fiscal year ending on October 12, 2023.",
            "output": {
                "reason": "The date in the answer contradicts the date in the ground_truth.",
                "verdict": "-1",
            },
        },
        {
            "question": "What human capital management initiatives has Disney reported on?",
            "answer": """Not enough information to tell.""",
            "ground_truth": "",
            "output": {
                "reason": "The answer indicates a lack of knowledge.",
                "verdict": "-1",
            },
        },
        {
            "question": "What are the theories about the possible influence of other pre-Columbian cultures on Machu Picchu's architecture?",
            "answer": """There is no specific information provided regarding the influence of other pre-Columbian cultures on Machu Picchu's architecture in the available sources. The sources discuss the architectural brilliance of the Incas, their use of polygonal masonry, and the integration of astronomical observations into their building practices, but do not mention direct influences from other pre-Columbian cultures.""",
            "ground_truth": "Pre-Columbian cultures had great influence on Machu Picchu's architecture.",
            "output": {
                "reason": "The **answer** indicates a lack of information and is unable to answer the question despite providing plausible discussion.",
                "verdict": "-1",
            },
        },
        {
            "question": "Is there a connection between the spiritual power of ancient sites and the economic development of surrounding areas?",
            "answer": "There is no direct information provided that establishes a clear connection between the spiritual power of ancient sites and the economic development of surrounding areas. However, it can be inferred that many of these ancient sites, such as Machu Picchu, the \"Cities of the Gods\", and Stonehenge, have become popular tourist attractions, which can contribute to the local economy.",
            "ground_truth": "Maybe",
            "output": {
                "reason": "The answer indicates a lack of information.",
                "verdict": -1,
            },
        },
        {
            "question": "What was Condoleezza Rice's view on the current global threats compared to the Cold War?",
            "answer": """Condoleezza Rice expressed that the current global threats differ significantly from those during the Cold War. She highlighted the importance of addressing contemporary challenges such as terrorism, the rise of authoritarian regimes, and the erosion of democratic values. Rice emphasized that the geopolitical landscape has evolved, with new forms of aggression and instability emerging, particularly from countries like Russia and China. She underscored the need for the United States and its allies to adapt their strategies to effectively counter these modern threats, which require a collaborative approach among democratic nations to uphold international order and security.""",
            "ground_truth": "Condoleezza Rice considered the emerging era to be more dangerous than the Cold War, citing the gathering global threats and the perils of 'the four horsemen of the Apocalypse\u2014populism, nativism, isolationism, and protectionism.",
            "output": {
                "reason": "The answer is overly general relative to the ground truth, providing general points on Condoleezza Rice's perspective on current versus Cold War threats without specifically capturing her view.",
                "verdict": "-1",
            },
        },
        {
            "question": "What was the name of the forum where Condoleezza Rice shared her views on global threats?",
            "answer": """The forum where Condoleezza Rice shared her views on global threats was the 9th Yalta Annual Meeting.""",
            "ground_truth": "The Atlantic Council Global Future Forum",
            "output": {
                "reason": "The question is underspecified, leading to multiple possible answers. Different answers could be factual depending on the time being asked about, which is not specified.",
                "verdict": "-1",
            },
        },
        {
            "question": "How did Condoleezza Rice's statements reflect her commitment to US democratic values?",
            "answer": """Condoleezza Rice's statements emphasize the importance of democracy, civil society, and the rule of law, reflecting her commitment to US democratic values. She advocates for the strengthening of democratic institutions in Ukraine, highlighting the need for a strong opposition and independent judiciary.""",
            "ground_truth": "Condoleezza Rice's statements reflected her commitment to US democratic values by emphasizing the importance of US leadership in the world and the need for the country to remain engaged globally. She warned against the dangers of populism, nativism, isolationism, and protectionism, and stressed that great powers like the US have a responsibility to shape the international environment.",
            "output": {
                "reason": "The answer and ground truth capture are phrased differently and the answer provides information not present in the ground truth, but they capture the SAME general idea that stronger democracy is needed and can be championed with US leadership and support.",
                "verdict": "1",
            },
        },
        {
            "question": "What human capital management initiatives has Disney reported on?",
            "answer": """Disney has implemented several key programs and initiatives for human capital management. These include DEI initiatives, health and wellness benefits, family resources, and other benefits, a continued response to COVID-19, the Disney Aspire education investment program, Talent Development programs, and a focus on Social Responsibility and Community. They also have environmental and sustainability goals.""",
            "ground_truth": "Diversity, Equity, and Inclusion (DE&I) initiatives, health, wellness, family resources.",
            "output": {
                "reason": "The answer and ground truth are consistent.",
                "verdict": "1",
            },
        },
        {
            "question": "What role did the local indigenous community play in the construction of Machu Picchu?",
            "answer": "The local indigenous community played a significant role in the construction of Machu Picchu, as they were likely involved in the labor required for its building. The Incas, who were skilled masons, utilized local resources and knowledge",
            "ground_truth": "The local indigenous community played a role in guiding Hiram Bingham to the ruins of Machu Picchu in 1911, but there is no information in the context about their role in the construction of Machu Picchu.",
            "output": {
                "reason": "The **ground truth** suggests lack of knowledge while the answer addresses the question. The answer introduces additional information not in the ground truth that is highly relevant and ON-FOCUS to answering the question.",
                "verdict": "0",
            },
        },
        {
            "question": "What human capital management initiatives has Disney reported on?",
            "answer": """Diversity, Equity, and Inclusion (DE&I) initiatives, health, wellness, family resources.""",
            "ground_truth": "",
            "output": {
                "reason": "The answer responds to the question.",
                "verdict": "0",
            },
        },
        {
            "question": "How did Condoleezza Rice think the US should engage with the world to promote democracy?",
            "answer": """Condoleezza Rice emphasized the importance of the United States maintaining a position of leadership. She highlighted that the U.S. should support countries like Ukraine in their efforts to strengthen democracy and resist authoritarian influences. Rice also pointed out that the U.S. cannot address global challenges alone and must work collaboratively with allies to promote democratic values and stability in regions facing threats to their governance.""",
            "ground_truth": "Condoleezza Rice emphasized the importance of the United States making a statement and reality of its willingness to remain engaged in the world, as great powers don't mind their own business and if the US doesn't shape the international environment, others like China and Russia will.",
            "output": {
                "reason": "The question is not underspecified yet both answers are plausible, capturing different dimensions of the same general idea that the US should maintain engagement as a leader.",
                "verdict": "0",
            },
        },
        {
            "question": "How did the Incas manage to build the site's thousands of stone steps?",
            "ground_truth": """The Incas built the site's thousands of stone steps, along with the rest of Machu Picchu, without the use of wheels or iron and steel tools, thanks to their expert building techniques.""",
            "answer": """The Incas built the thousands of stone steps at Machu Picchu by utilizing the geological features of the site, specifically the network of tectonic faults beneath it. The orientation of the steps and buildings was aligned with these geological features, which not only facilitated construction but also helped with drainage during heavy rain.""",
            "output": {
                "reason": "The answer provides a more detailed explanation of the construction process. Although it omits key points from the ground truth, such as the lack of wheels, the answer is very much plausible and thus both are viable responses to the question.",
                "verdict": "0",
            },
        }
    ],
    input_keys=["question", "answer", "ground_truth"],
    output_key="output",
    output_type="json",
    language="english",
)
