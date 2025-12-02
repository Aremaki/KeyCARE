from openai import OpenAI
import json
from tqdm import tqdm
from .Relator import Relator  # same as your base class


class LLMRelator(Relator):
    """
    LLM-based judge classifier, drop-in replacement for TransformersRelator.
    """

    def __init__(
        self,
        batch_size=8,
        model="gpt-4.1-mini",
        temperature=0.0,
    ):
        self.client = OpenAI()
        self.batch_size = batch_size
        self.model = model
        self.temperature = temperature

        # fixed allowed labels from your base class
        self.allowed_labels = ["NO_RELATION", "EXACT", "BROAD", "NARROW"]

    def _build_prompt(self, source_concept: dict, target_concept: dict):
        """
        Build the LLM prompt for one source-target pair.
        s and t are objects with `text`, `code`, `title`, `synonyms`
        """

        prompt = f"""
You are an expert biomedical ontology curator. 
Your task is to determine the semantic relationship between two concepts.

Possible labels:
- EXACT: Both concepts refer to the same medical idea.
- BROAD: The SOURCE concept is more general than the TARGET concept.
- NARROW: The SOURCE concept is more specific than the TARGET concept.
- NO_RELATION: The two concepts describe different medical ideas.

Rules:
- Use ONLY the provided titles, codes, and synonyms.
- Do NOT hallucinate missing information.
- Output ONLY a JSON dictionary: {{"label": "<LABEL>"}}

### SOURCE CONCEPT
Code: {source_concept["code"]}
Title: {source_concept["title"]}
Synonyms: {", ".join(source_concept["synonyms"]) if source_concept["synonyms"] else "None"}

### TARGET CONCEPT
Code: {target_concept["code"]}
Title: {target_concept["title"]}
Synonyms: {", ".join(target_concept["synonyms"]) if target_concept["synonyms"] else "None"}

Return only: {{"label": "<LABEL>"}}
"""
        return prompt

    def _call_llm(self, prompts):
        """
        Batch call the LLM with multiple prompts.
        prompts: list of strings
        returns: list of labels
        """

        outputs = []
        for prompt in prompts:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict biomedical concept relation classifier.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content

            try:
                parsed = json.loads(raw)
                label = parsed.get("label", "NO_RELATION").upper()
            except Exception:
                print(f"Error parsing response: {raw}")
                label = "NO_RELATION"

            if label not in self.allowed_labels:
                label = "NO_RELATION"

            outputs.append(label)

        return outputs

    def compute_relation(self, source: list[dict], target: list[dict]):
        """
        Inputs:
           source = list of concept objects
           target = list of concept objects
        Output:
           list[list[str]] where each element is like ["BROAD"] or ["NO_RELATION"]
        """

        final_labels = [None] * len(source)

        valid_indices = []
        valid_pairs = []

        # Collect valid pairs
        for i, (source_concept, target_concept) in enumerate(zip(source, target)):
            if source_concept["text"] and target_concept["text"]:
                valid_indices.append(i)
                valid_pairs.append((source_concept, target_concept))
            else:
                final_labels[i] = ["NO_RELATION"]  # type: ignore

        if not valid_pairs:
            return final_labels

        # Batch processing
        for b in tqdm(
            range(0, len(valid_pairs), self.batch_size), desc="LLM Judge Batches"
        ):
            batch_pairs = valid_pairs[b : b + self.batch_size]

            prompts = [
                self._build_prompt(source_concept, target_concept)
                for source_concept, target_concept in batch_pairs
            ]
            batch_labels = self._call_llm(prompts)

            # assign labels back to correct indices
            for label in batch_labels:
                idx = valid_indices.pop(0)
                final_labels[idx] = [label]

        return final_labels
