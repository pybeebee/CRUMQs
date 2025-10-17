import sys
import json
import re
from collections import defaultdict
from typing import Any, List, Optional
from tqdm import tqdm
from termcolor import colored
import json_repair 

from src.crumqs_generation.utils_inference import *
from sentence_transformers import SentenceTransformer

sys.path.append("..")

from src.crumqs_generation.prompts.deduplication import SYSTEM_PROMPT as IDENTIFY_PARAPHRASE_SYSTEM
from src.crumqs_generation.prompts.deduplication import USER_PROMPT as IDENTIFY_PARAPHRASE_USER

def safe_extract_output(output, placeholder_response, key=None):
    try:
        if key:
            try: 
                return json.loads("{"+output+"}")[key]
            except:
                return json.loads(output)[key]
        else: 
            try:
                return json.loads("{"+output+"}")
            except:
                return json.loads(output)
    except Exception:
        print(colored("Invalid response, setting to default result:", "yellow"), output)
        return placeholder_response

def strip_ansi(text):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

def custom_metadata(record: dict, _: int):
    return {
        "source": record.get("source"),
        "doc_id": record.get("doc_id"),
        "rr_id": record.get("rr_id"),
        "topic": record.get("topic"),
    }

def log(text, logging_file, mode='a'):
    print(text)
    with open(logging_file, mode) as f:
        if ">>>" in strip_ansi(str(text)):
            f.write("\n\n"+"#"*20+strip_ansi(str(text)))
        else:
            f.write("\n"+strip_ansi(str(text)))

### HELPER: Process prompts in batches to avoid overloading LiteLLM
async def get_response_batch_chunked(system_prompt, user_prompts, provider, temperature, use_cache, batch_size=100, max_tokens=500):

    if len(user_prompts) <= batch_size:
        # If prompts are within batch size, process normally
        return await get_response_batch(
            system_prompt=system_prompt,
            user_prompts=user_prompts,
            provider=provider,
            temperature=temperature,
            use_cache=use_cache,
            max_tokens=max_tokens,
        )
    
    # Process in chunks
    all_outputs = []
    for i in tqdm(range(0, len(user_prompts), batch_size), desc="Processing prompt batches"):
        batch_prompts = user_prompts[i:i + batch_size]
        
        batch_outputs = await get_response_batch(
            system_prompt=system_prompt,
            user_prompts=batch_prompts,
            max_tokens=max_tokens,
            provider=provider,
            temperature=temperature,
            use_cache=use_cache,
        )
        all_outputs.extend(batch_outputs)
    
    return all_outputs

def json_loads(json_str: str) -> Any:
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    elif json_str.startswith("```"):
        json_str = json_str[3:]
    if "```" in json_str:  # ending  ```json
        json_str = json_str.split("```")[0].strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"Failed to decode JSON: {json_str}")

        try:
            x = json_repair.repair_json(json_str, return_objects=True)
            if "response" not in x or "explanation" not in x:
                raise ValueError
        except: 
            return {"response": "", "explanation": ""}
        return {"response": "", "explanation": ""}


def group_items(pairs):
    """Union-Find algorithm to group items based on pairs of connections."""
    parent = {}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # path compression
        return parent[x]

    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    # Initialize parent for each item
    for a, b in pairs:
        if a not in parent:
            parent[a] = a
        if b not in parent:
            parent[b] = b
        union(a, b)

    # Grouping by root
    groups = {}
    for item in parent:
        root = find(item)
        if root not in groups:
            groups[root] = []
        groups[root].append(item)

    # Assign group numbers
    group_mapping = {}
    for i, group_items in enumerate(groups.values()):
        for item in group_items:
            group_mapping[item] = i

    return group_mapping, i


def invert_group_mapping(group_mapping):
    inverse = defaultdict(list)
    for item, group in group_mapping.items():
        inverse[group].append(item)
    return dict(inverse)


class FindPairs:
    PairsList = List[dict[str, Any]]

    def __init__(self, embedding_model: str, similarity_threshold: float = 0.8):
        """Initialize the FindPairs class with embedding model and similarity threshold."""
        self.encoder = SentenceTransformer(embedding_model)
        self.threshold = similarity_threshold
        self.candidate_pairs_d = {}

    def find_similar_pairs(
        self, texts1: List[str], request_id: str, texts2: Optional[List[str]] = None
    ):
        """Find question pairs with similarity above threshold using embeddings.

        Args:
            texts1: First list of texts
            request_id: Identifier for this request
            texts2: Optional second list of texts. If provided, compares texts1 vs texts2.
                   If None, compares texts1 vs texts1 (self-similarity).
        """
        # Set up for self-similarity or cross-similarity
        if texts2 is None:
            texts2 = texts1
            embeddings1 = self.encoder.encode(texts1)
            embeddings2 = embeddings1
            similarity_matrix = self.encoder.similarity(embeddings1, embeddings2)
            is_self_similarity = True
        else:
            embeddings1 = self.encoder.encode(texts1)
            embeddings2 = self.encoder.encode(texts2)
            similarity_matrix = self.encoder.similarity(embeddings1, embeddings2)
            is_self_similarity = False

        # Find pairs above threshold
        similar_pairs = []
        for i in range(len(texts1)):
            # For self-similarity, start from i+1 to avoid duplicates and self-matches
            # For cross-similarity, compare all pairs
            j_start = i + 1 if is_self_similarity else 0
            for j in range(j_start, len(texts2)):
                if similarity_matrix[i][j] > self.threshold:
                    similar_pairs.append(
                        {
                            "q1_idx": i,
                            "q2_idx": j,
                            "q1": texts1[i],
                            "q2": texts2[j],
                            "similarity": float(
                                similarity_matrix[i][j]
                            ),  # Convert to float for JSON serialization
                        }
                    )

        self.candidate_pairs_d[request_id] = similar_pairs

    async def check_pairs_with_llm(self, request_id: str):
        """Use LLM to verify if question pairs are actually paraphrases."""
        # Prepare prompts for all pairs
        user_prompts = []
        candidate_pairs = self.candidate_pairs_d[request_id]

        for pair in candidate_pairs:
            question_a = pair["q1"]
            question_b = pair["q2"]
            user_prompt = IDENTIFY_PARAPHRASE_USER.format(
                question_a=question_a,
                question_b=question_b,
            )
            user_prompts.append(user_prompt)

        # Get LLM responses for all pairs using batch processing
        responses = await get_response_batch(
            system_prompt=IDENTIFY_PARAPHRASE_SYSTEM,
            user_prompts=user_prompts,
            provider=ModelProvider.LITELLM_LOCAL,
            stop=["<|eot_id|>"],
        )

        # Process responses and add LLM decisions to pairs
        for pair, response in zip(candidate_pairs, responses):
            llm_response = json_loads(response)
            pair["llm_choice"] = llm_response["response"] == "YES"
            pair["llm_explanation"] = llm_response["explanation"]

    def get_pairs(self, request_id: str, verified_only: bool = False) -> PairsList:
        if request_id not in self.candidate_pairs_d:
            raise ValueError(f"No candidate pairs found for request_id: {request_id}")
        pairs = self.candidate_pairs_d[request_id]
        if verified_only:
            # Filter pairs where LLM confirmed they are paraphrases
            return [pair for pair in pairs if pair.get("llm_choice", True)]
        return pairs

    def get_pairs_d(self, verified_only: bool = False) -> dict[str, PairsList]:
        """Get all pairs across all requests, optionally filtering by LLM verification."""
        pairs_d = self.candidate_pairs_d
        if verified_only:
            return {
                request_id: self.get_pairs(request_id, verified_only=True) for request_id in pairs_d
            }
        return pairs_d

    @staticmethod
    def get_best_unique_matching(pairs: PairsList) -> PairsList:
        """Get the best unique matching where each question appears at most once.

        Uses a greedy approach with augmenting paths to find near-optimal matching
        that maximizes total similarity while ensuring each question appears at most once.

        Args:
            request_id: Identifier for the request
            verified_only: If True, only consider LLM-verified pairs

        Returns:
            List of pairs where each q1 and q2 appears at most once, optimally matched
        """

        if not pairs:
            return []

        # Sort pairs by similarity (highest first) with tie-breaking on q1 and q2
        sorted_pairs = sorted(pairs, key=lambda x: (-x["similarity"], x["q1"], x["q2"]))

        # Track which questions are already matched
        matched = set()
        result = []

        # Build adjacency list with all possible connections
        graph = {}
        pair_lookup = {}  # For quick pair object lookup
        for pair in pairs:
            q1, q2 = pair["q1"], pair["q2"]
            if q1 not in graph:
                graph[q1] = []
            if q2 not in graph:
                graph[q2] = []
            graph[q1].append((q2, pair["similarity"], pair))
            graph[q2].append((q1, pair["similarity"], pair))
            # Store pair objects for both directions
            pair_lookup[(q1, q2)] = pair
            pair_lookup[(q2, q1)] = pair

        # Greedy matching
        for pair in sorted_pairs:
            q1, q2 = pair["q1"], pair["q2"]
            if q1 not in matched and q2 not in matched:
                matched.add(q1)
                matched.add(q2)
                result.append(pair)

        # Try to improve by finding augmenting paths
        # This handles cases where greedy isn't optimal (like the A-B-Z-C example)
        improved = True
        while improved:
            improved = False

            # Find unmatched questions
            all_questions = set(graph.keys())
            unmatched = all_questions - matched

            for unmatched_q in unmatched:
                # Try to find an augmenting path that improves total weight
                if FindPairs._find_augmenting_path(unmatched_q, graph, matched, result):
                    improved = True
                    break

        return result

    @staticmethod
    def _find_augmenting_path(start_q, graph, matched, current_matching):
        """Find an augmenting path starting from an unmatched question."""
        # Get current matching as dict for quick lookup
        match_dict = {}
        for pair in current_matching:
            q1, q2 = pair["q1"], pair["q2"]
            match_dict[q1] = q2
            match_dict[q2] = q1

        # Ensure consistent neighbor iteration by sorting neighbors
        for neighbor, weight, pair_obj in sorted(
            graph.get(start_q, []), key=lambda x: (x[1], x[0])
        ):
            if neighbor not in matched:
                # Direct connection to unmatched - always beneficial
                matched.add(start_q)
                matched.add(neighbor)
                current_matching.append(pair_obj)
                return True

            # neighbor is matched - see if we can improve by "stealing" it
            current_partner = match_dict[neighbor]

            # Find the current pair involving this neighbor
            current_pair = None
            current_weight = 0
            for i, pair in enumerate(current_matching):
                if (pair["q1"] == neighbor and pair["q2"] == current_partner) or (
                    pair["q1"] == current_partner and pair["q2"] == neighbor
                ):
                    current_pair = (i, pair)
                    current_weight = pair["similarity"]
                    break

            if current_pair is None:
                continue

            # Check if the displaced partner has better alternatives
            best_alternative_weight = 0
            best_alternative_pair = None

            for alt_neighbor, alt_weight, alt_pair in graph.get(current_partner, []):
                if alt_neighbor not in matched and alt_neighbor != start_q:
                    # Found an unmatched alternative
                    if alt_weight > best_alternative_weight:
                        best_alternative_weight = alt_weight
                        best_alternative_pair = alt_pair

            # Check if the swap improves total weight
            # New total: weight (start_q-neighbor) + best_alternative_weight (current_partner-alternative)
            # Old total: current_weight (neighbor-current_partner)
            if best_alternative_pair and (weight + best_alternative_weight) > current_weight:
                # Perform the beneficial swap
                # Remove old matching
                current_matching.pop(current_pair[0])
                matched.remove(current_partner)

                # Add new matchings
                matched.add(start_q)
                current_matching.append(pair_obj)  # start_q-neighbor

                # Match the displaced partner with its best alternative
                alt_neighbor = (
                    best_alternative_pair["q1"]
                    if best_alternative_pair["q1"] != current_partner
                    else best_alternative_pair["q2"]
                )
                matched.add(alt_neighbor)
                current_matching.append(best_alternative_pair)

                return True

        return False

