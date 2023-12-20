"""
The code in this file works to train a lemma classifier for 's
"""

import torch 
import torch.nn as nn
import torch.optim as optim
import os 
import logging
import argparse
from os import path
from os import remove
from typing import List, Tuple, Any

from stanza.models.common.foundation_cache import load_pretrain
from stanza.models.lemma_classifier import utils
from stanza.models.lemma_classifier.model import LemmaClassifier
from stanza.utils.get_tqdm import get_tqdm

tqdm = get_tqdm()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LemmaClassifierTrainer():
    """
    Class to assist with training a LemmaClassifier
    """

    def __init__(self, vocab_size: int, embedding_file: str, embedding_dim: int, hidden_dim: int, output_dim: int, use_charlm: bool, **kwargs):
        """
        Initializes the LemmaClassifierTrainer class.
        
        Args:
            vocab_size (int): Size of the vocab being used (if custom vocab)
            embedding_file (str): What word embeddings file to use.  Use a Stanza pretrain .pt
            embedding_dim (int): Size of embedding dimension to use on the aforementioned word embeddings
            hidden_dim (int): Size of hidden vectors in LSTM layers
            output_dim (int): Size of output vector from MLP layer
            use_charlm (bool): Whether to use charlm embeddings as well

        Kwargs:
            forward_charlm_file (str): Path to the forward pass embeddings for the charlm 
            backward_charlm_file (str): Path to the backward pass embeddings for the charlm
            lr (float): Learning rate, defaults to 0.001.
            loss_func (str): Which loss function to use (either 'ce' or 'weighted_bce') 

        Raises:
            FileNotFoundError: If the forward charlm file is not present
            FileNotFoundError: If the backward charlm file is not present
        """
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Load word embeddings
        pt = load_pretrain(embedding_file)
        emb_matrix = pt.emb
        # TODO: could refactor only the trained embeddings, then turn freezing back on, then don't save the full PT with the model
        self.embeddings = nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=False)
        self.embeddings.weight.requires_grad = True
        self.vocab_map = { word.replace('\xa0', ' '): i for i, word in enumerate(pt.vocab) }
        self.vocab_size = emb_matrix.shape[0]
        self.embedding_dim = emb_matrix.shape[1]

        # Load CharLM embeddings
        forward_charlm_file = kwargs.get("forward_charlm_file")
        backward_charlm_file = kwargs.get("backward_charlm_file")
        if use_charlm and forward_charlm_file is not None and not os.path.exists(forward_charlm_file):
            raise FileNotFoundError(f"Could not find forward charlm file: {forward_charlm_file}")
        if use_charlm and backward_charlm_file is not None and not os.path.exists(backward_charlm_file):
            raise FileNotFoundError(f"Could not find backward charlm file: {backward_charlm_file}")

        # TODO: embedding_dim and vocab_size are read off the embeddings file
        self.model = LemmaClassifier(self.vocab_size, self.embedding_dim, hidden_dim, output_dim, self.vocab_map, self.embeddings, charlm=use_charlm,
                                     charlm_forward_file=forward_charlm_file, charlm_backward_file=backward_charlm_file)
        
        # Find loss function
        loss_fn = kwargs.get("loss_func", "ce").lower() 
        if loss_fn == "ce":
            self.criterion = nn.CrossEntropyLoss()
            self.weighted_loss = False
        elif loss_fn == "weighted_bce":
            self.criteron = nn.BCEWithLogitsLoss()  
            self.weighted_loss = True  # used to add weights during train time.
        else:
            raise ValueError("Must enter a valid loss function (e.g. 'ce' or 'weighted_bce')")

        self.optimizer = optim.Adam(self.model.parameters(), lr=kwargs.get("lr", 0.001))  

    def train(self, texts_batch: List[List[str]], positions_batch: List[int], labels_batch: List[int], num_epochs: int, save_name: str, **kwargs) -> None:

        """
        Trains a model on batches of texts, position indices of the target token, and labels (lemma annotation) for the target token.

        Args:
            texts_batch (List[List[str]]): Batches of tokenized texts, one per sentence. Expected to contain at least one instance of the target token.
            positions_batch (List[int]): Batches of position indices (zero-indexed) for the target token, one per input sentence. 
            labels_batch (List[int]): Batches of labels for the target token, one per input sentence. 
            num_epochs (int): Number of training epochs
            save_name (str): Path to file where trained model should be saved. 

        Kwargs:
            train_path (str): Path to data file, containing tokenized text sentences, token index and true label for token lemma on each line.         
        """

        train_path = kwargs.get("train_path")
        if train_path:  # use file to train model
            texts_batch, positions_batch, labels_batch, counts, label_decoder = utils.load_dataset(train_path, get_counts=self.weighted_loss)
            logging.info(f"Loaded dataset successfully from {train_path}")
            logging.info(f"Using label decoder: {label_decoder}")

        assert len(texts_batch) == len(positions_batch) == len(labels_batch), f"Input batch sizes did not match ({len(texts_batch)}, {len(positions_batch)}, {len(labels_batch)})."
        if path.exists(save_name):
            raise FileExistsError(f"Save name {save_name} already exists; training would overwrite previous file contents. Aborting...")
        
        # Configure weighted loss, if necessary
        if self.weighted_loss:
            weights = [0 for _ in label_decoder.keys()]  # each key in the label decoder is one class, we have one weight per class
            total_samples = sum(counts.values())
            for class_idx in counts:
                weights[class_idx] = total_samples / (counts[class_idx] * len(counts))  # weight_i = total / (# examples in class i * num classes)
                weights = torch.tensor(weights)
            logging.info(f"Using weights {weights} for weighted loss.")
            self.criterion = nn.BCEWithLogitsLoss(weight=weights)

        logging.info("Embedding norm: %s", torch.linalg.norm(self.model.embedding.weight))
        for epoch in range(num_epochs):
            # go over entire dataset with each epoch
            for texts, position, label in tqdm(zip(texts_batch, positions_batch, labels_batch), total=len(texts_batch)):
                if position < 0 or position > len(texts) - 1:  # validate position index
                    raise ValueError(f"Found position {position} in text: {texts}, which is not possible.")
                
                self.optimizer.zero_grad()

                output = self.model(position, texts)
                
                # Compute loss, which is different if using CE or BCEWithLogitsLoss
                if self.weighted_loss:  # BCEWithLogitsLoss requires a vector for target where probability is 1 on the true label class, and 0 on others.
                    target_vec = [1, 0] if label == 0 else [0, 1]
                    target = torch.tensor(target_vec, dtype=torch.float32)
                else:  # CELoss accepts target as just raw label
                    target = torch.tensor(label, dtype=torch.long)
                loss = self.criterion(output, target)

                loss.backward()
                self.optimizer.step()
            
            logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")
            logging.info("Embedding norm: %s", torch.linalg.norm(self.model.embedding.weight))

        save_dir = os.path.split(save_name)[0]
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        state_dict = {
            "params": self.model.state_dict(),
            "label_decoder": label_decoder,
        }
        torch.save(state_dict, save_name)
        logging.info(f"Saved model state dict to {save_name}")

def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10000, help="Number of tokens in vocab")
    parser.add_argument("--embedding_dim", type=int, default=100, help="Number of dimensions in word embeddings (currently using GloVe)")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Size of hidden layer")
    parser.add_argument("--output_dim", type=int, default=2, help="Size of output layer (number of classes)")
    parser.add_argument('--wordvec_pretrain_file', type=str, default=None, help='Exact name of the pretrain file to read')
    parser.add_argument("--charlm", action='store_true', dest='use_charlm', default=False, help="Whether not to use the charlm embeddings")
    parser.add_argument('--charlm_shorthand', type=str, default=None, help="Shorthand for character-level language model training corpus.")
    parser.add_argument("--charlm_forward_file", type=str, default=os.path.join(os.path.dirname(__file__), "charlm_files", "1billion_forward.pt"), help="Path to forward charlm file")
    parser.add_argument("--charlm_backward_file", type=str, default=os.path.join(os.path.dirname(__file__), "charlm_files", "1billion_backwards.pt"), help="Path to backward charlm file")
    parser.add_argument("--save_name", type=str, default=path.join(path.dirname(__file__), "saved_models", "lemma_classifier_model_weighted_loss_charlm_new.pt"), help="Path to model save file")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--num_epochs", type=float, default=10, help="Number of training epochs")
    parser.add_argument("--train_file", type=str, default=os.path.join(os.path.dirname(__file__), "test_sets", "combined_train.txt"), help="Full path to training file")
    parser.add_argument("--weighted_loss", action='store_true', dest='weighted_loss', default=False, help="Whether to use weighted loss during training.")
    return parser

def main(args=None):
    parser = build_argparse()
    args = parser.parse_args(args)

    vocab_size = args.vocab_size
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    wordvec_pretrain_file = args.wordvec_pretrain_file
    use_charlm = args.use_charlm
    forward_charlm_file = args.charlm_forward_file
    backward_charlm_file = args.charlm_backward_file
    save_name = args.save_name 
    lr = args.lr
    num_epochs = args.num_epochs
    train_file = args.train_file
    weighted_loss = args.weighted_loss

    if os.path.exists(save_name):
        raise FileExistsError(f"Save name {save_name} already exists. Training would override existing data. Aborting...")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file {train_file} not found. Try again with a valid path.")

    logging.info("Running training script with the following args:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    logging.info("------------------------------------------------------------")

    trainer = LemmaClassifierTrainer(vocab_size=vocab_size,
                                     embedding_file=wordvec_pretrain_file,
                                     embedding_dim=embedding_dim,
                                     hidden_dim=hidden_dim,
                                     output_dim=output_dim,
                                     use_charlm=use_charlm,
                                     forward_charlm_file=forward_charlm_file,
                                     backward_charlm_file=backward_charlm_file,
                                     lr=lr,
                                     loss_func="weighted_bce" if weighted_loss else "ce"
                                     )

    trainer.train(
        [], [], [], num_epochs=num_epochs, save_name=save_name, train_path=train_file
    )

if __name__ == "__main__":
    main()

