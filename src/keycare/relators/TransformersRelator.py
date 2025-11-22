import torch
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .Relator import Relator


class TransformersRelator(Relator):
    def __init__(self, n, threshold, model_path, batch_size=32):
        """
        Initializes TransformersRelator, a class inheriting from Relator.

        Parameters:
        n (int): Maximum number of labels for a single relation.
        threshold (float): Threshold value used for mentions relation.
        model_path (str): Path to the model class.
        batch_size (int): Batch size for processing relations.
        """
        super().__init__(n, threshold, model_path)
        self.batch_size = batch_size

    def initialize_pretrained_model(self, model_path):
        """
        Initializes a pretrained model for TransformersRelator based on the provided model_path.

        Parameters:
        model_path (str): Path to the pretrained model.

        Returns:
        object: Pretrained model instance.
        """
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.labels])
        path = "BSC-NLP4BIA/SapBERT-from-roberta-base-biomedical-clinical-es"
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if model_path is None:
            path = "BSC-NLP4BIA/biomedical-semantic-relation-classifier"
            model = AutoModelForSequenceClassification.from_pretrained(path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return model

    def compute_relation(self, source, target):
        """
        Computes relations between source and target entities using Transformers model.

        Parameters:
        source (list): List of source entities.
        target (list): List of target entities.

        Returns:
        list: List of labels representing computed relations.
        """
        final_labels: list[list[str] | None] = [None] * len(source)

        valid_indices = []
        valid_source_text = []
        valid_target_text = []

        for i, (s, t) in enumerate(zip(source, target)):
            if s.text and t.text:
                valid_indices.append(i)
                valid_source_text.append(s.text)
                valid_target_text.append(t.text)
            else:
                final_labels[i] = ["NO_RELATION"]

        if not valid_indices:
            return final_labels

        all_logits = []
        for i in tqdm(
            range(0, len(valid_source_text), self.batch_size), desc="Processing batches"
        ):
            batch_source = valid_source_text[i : i + self.batch_size]
            batch_target = valid_target_text[i : i + self.batch_size]

            tokenized_mention = self.tokenizer(
                batch_source,
                batch_target,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            with torch.no_grad():
                output = self.model(**tokenized_mention)

            all_logits.append(output.logits)

        logits = torch.cat(all_logits, dim=0)

        for i, logit in enumerate(tqdm(logits, desc="Computing relations")):
            original_index = valid_indices[i]
            predscores = {label: score for label, score in zip(self.labels, logit)}
            top_n_labels = sorted(
                predscores, key=lambda label: predscores[label], reverse=True
            )[: self.n]
            filtered_labels = [
                label for label in top_n_labels if predscores[label] > self.threshold
            ]
            final_labels[original_index] = filtered_labels

        return final_labels
