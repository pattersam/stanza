import torch
import torch.nn as nn
import os
import sys
import logging

from transformers import AutoTokenizer, AutoModel
from typing import Mapping, List, Tuple, Any

from stanza.models.lemma_classifier.base_model import LemmaClassifier
from stanza.models.lemma_classifier.constants import ModelType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LemmaClassifierWithTransformer(LemmaClassifier):
    def __init__(self, output_dim: int, transformer_name: str, label_decoder: Mapping):
        """
        Model architecture:

            Use a transformer (BERT or RoBERTa) to extract contextual embedding over a sentence.
            Get the embedding for the word that is to be classified on, and feed the embedding
            as input to an MLP classifier that has 2 linear layers, and a prediction head.

        Args:
            output_dim (int): Dimension of the output from the MLP 
            model_type (str): What kind of transformer to use for embeddings ('bert' or 'roberta')
        """
        super(LemmaClassifierWithTransformer, self).__init__()

        # Choose transformer
        self.transformer_name = transformer_name
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        self.transformer = AutoModel.from_pretrained(transformer_name)
        config = self.transformer.config

        embedding_size = config.hidden_size

        # define an MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.label_decoder = label_decoder

    def forward(self, pos_index: int, text: List[str]):
        """

        Args:
            text (List[str]): A single sentence with each token as an entry in the list.
            pos_index (int): The index of the token to classify on.

        Returns the logits of the MLP
        """

        # Get the transformer embeddings 
        input_ids = self.tokenizer.convert_tokens_to_ids(text)

        # Convert tokens to IDs and put them into a tensor
        input_ids_tensor = torch.tensor([input_ids], device=next(self.parameters()).device)  # move data to device as well
        # Forward pass through Transformer
        with torch.no_grad():
            outputs = self.transformer(input_ids_tensor)
        
        # Get embeddings for all tokens
        last_hidden_state = outputs.last_hidden_state
        token_embeddings = last_hidden_state[0]

        pos_index = torch.tensor(pos_index, device=next(self.parameters()).device)
        # Get target embedding
        target_pos_embedding = token_embeddings[pos_index]

        # pass to the MLP
        output = self.mlp(target_pos_embedding)
        return output

    def model_type(self):
        return ModelType.TRANSFORMER
