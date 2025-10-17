claim_extraction = """A "claim" is a statement or assertion made within a text expressing a belief, opinion, or fact. Given the context, please extract all claims.

Note: The claims should not contain ambiguous references, such as ’he’,’ she,’ and’ it’, and should use complete names. You should substitute pronouns or incomplete names with the specific subject or object being referred to and, if needed, incorporate the most important distinguishing details such as location/profession/time-period to distinguish from others who might share similar names. Make sure to extract claims based on the given context; don’t generate the evidence or claims yourself. If there is no claim present, respond with the empty list. Provide the claims as a list of strings, followed by "=====". DO NOT produce any other text.

For example, the response should follow this format:
["claim1", "claim2", "claim3", ...]=====

Context:
{document}

Claims:"""


triple_extraction = """Extract content graph with sentences in forms of triples (entity, relation, entity) based only on the provided sentences. The triples should not contain ambiguous references, such as ’he’,’ she,’ and’ it’, and should use complete names. Provide the triples as a list of tuples of strings, followed by "=====". DO NOT produce any other text.

Examples:

Provided Sentences:
Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer. Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director.

List of Triples in Provided Sentences:
[("Scott Derrickson", "born on", "July 16, 1966"), ("Scott Derrickson", "is", "American"), ("Scott Derrickson", "is", "director"), ("Scott Derrickson", "is", "screenwriter"), ("Scott Derrickson", "is", "producer"), ("Edward Davis Wood Jr.", "born on", "October 10, 1924"), ("Edward Davis Wood Jr.", "died on", "December 10, 1978"), ("Edward Davis Wood Jr.", "was", "American"), ("Edward Davis Wood Jr.", "was", "filmmaker"), ("Edward Davis Wood Jr.", "was", "actor"), ("Edward Davis Wood Jr.", "was", "writer"), ("Edward Davis Wood Jr.", "was", "producer"), ("Edward Davis Wood Jr.", "was", "director")]=====

Your turn:

Provided Sentences:
{document}

List of Triples in Provided Sentences:"""


complex_triple_extraction = """Given a news article, go over every sentence and extract triples in forms of ("entity", "entity", "a short description about the relation between two entities"). The triples should not contain ambiguous references, such as ’he’,’ she,’ and’ it’, and should use complete names. Group the triples with the same entity into a single list. Respond with a list of these component lists. Conclude your answer with "=====". DO NOT produce any other text.

Examples:

Provided Sentences:
Hunt also said the government needs to reform the welfare system to get more people back to work. The number of people not in the workforce for physical or mental health reasons has soared since the pandemic. Ken Clarke, a former Conservative Treasury chief, said cutting inheritance tax "might appeal to the Conservative right, but it leaves them open to the most appalling criticisms when inflation and the state of affairs is making poorer people in this country very vulnerable indeed." "I’m not sure that the economic and financial state of the country justifies it."  

List of List(s) of Triples in Provided Sentences:
[[("Hunt", "government", "Hunt said something about the government"), ("government", "welfare system", "government need to reform welfare system"), ("welfare system", "people", "reformed welfare system can get people back to work"), ("people", "physical or mental health reasons", "number of people not at work has soared due to physical or mental health reasons")], [("Ken Clarke", "former Conservative Treasury chief", "Ken Clarke is former Conservative Treasury chief"), ("Ken Clarke", "cutting inheritance tax", "Ken Clarke said something about cutting inheritance tax"), ("cutting inheritance tax", "Conservative right", "cutting inheritance tax appeal to the Conservative right"), ("cutting inheritance tax", "criticisms", "cutting inheritance tax leaves open to criticisms"), ("cutting inheritance tax", "inflation and the state of affairs", "cutting inheritance tax cause inflation and the state of affairs"), ("inflation and the state of affairs", "poorer people", "inflation and the state of affairs make poorer people vulnerable"), ("economic and financial state of the country", "cutting inheritance tax", "economic and financial state of the country might not justify cutting inheritance tax")]=====

Your turn:

Provided Sentences:
{document}

List of List(s) of Triples in Provided Sentences:"""