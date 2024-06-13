import argparse
import logging
import os 
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer

# To add Stanza modules
ROOT = '/Users/alexshan/Desktop/stanza'
sys.path.append(ROOT)

from stanza.models.common.utils import default_device
from stanza.models.common.foundation_cache import load_pretrain
from stanza.models.summarization.constants import * 
from stanza.models.summarization.src.model import *
from stanza.utils.get_tqdm import get_tqdm
from stanza.models.summarization.src.utils import *
from stanza.models.summarization.src.prepare_dataset import Dataset
from stanza.models.summarization.src.evaluate_model import evaluate_rouge_from_path

from typing import List, Tuple, Any, Mapping

torch.set_printoptions(threshold=100, edgeitems=5, linewidth=100)
logger = logging.getLogger('stanza.summarization') 
logger.propagate = False

# Check if the logger has handlers already configured
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

tqdm = get_tqdm()


class SummarizationTrainer():

    def __init__(self, model_args: dict, embedding_file: str, lr: float):
        """
        Model arguments:
        {
        batch_size (int): size of data batches used during training, 
        enc_hidden_dim (int): Size of encoder hidden state,
        enc_num_layers (int): Number of layers in the encoder LSTM,
        dec_hidden_dim (int): Size of decoder hidden state,
        dec_num_layers (int): Number of layers in the decoder LSTM,
        pgen (bool): Whether to use the pointergen feature in the model,
        coverage (bool): Whether to include coverage vectors in the decoder,
        }

        embedding_file (str): Path to the word vector pretrain file for embedding layer
        lr (float): Learning rate during training
        """
        self.model_args = model_args

        pt = load_pretrain(embedding_file)
        self.pt_embedding = pt
        self.lr = lr
        self.device = default_device() 
        self.max_enc_steps = self.model_args.get("max_enc_steps", None)
        self.max_dec_steps = self.model_args.get("max_dec_steps", None)

    def build_model(self) -> BaselineSeq2Seq:
        """
        Build the model for training using the model args

        Raises any errors depending on model argument errors
        """

        # parse input for valid args
        batch_size = self.model_args.get("batch_size", DEFAULT_BATCH_SIZE)
        encoder_hidden_dim = self.model_args.get("enc_hidden_dim", DEFAULT_ENCODER_HIDDEN_DIM)
        encoder_num_layers = self.model_args.get("enc_num_layers", DEFAULT_ENCODER_NUM_LAYERS)
        decoder_hidden_dim = self.model_args.get("dec_hidden_dim", DEFAULT_DECODER_HIDDEN_DIM)
        decoder_num_layers = self.model_args.get("dec_num_layers", DEFAULT_DECODER_NUM_LAYERS)
        pgen = self.model_args.get("pgen", False)
        coverage = self.model_args.get("coverage", False)
        use_charlm = self.model_args.get("charlm", False)
        charlm_forward_file = self.model_args.get("charlm_forward_file", None)
        charlm_backward_file = self.model_args.get("charlm_backward_file", None)
        max_enc_steps = self.model_args.get("max_enc_steps", None)
        max_dec_steps = self.model_args.get("max_dec_steps", None)

        parsed_model_args = {
            "batch_size": batch_size,
            "encoder_hidden_dim": encoder_hidden_dim,
            "encoder_num_layers": encoder_num_layers,
            "decoder_hidden_dim": decoder_hidden_dim,
            "decoder_num_layers": decoder_num_layers,
            "pgen": pgen,
            "coverage": coverage,
            "max_enc_steps": max_enc_steps,
            "max_dec_steps": max_dec_steps,
        }

        # return the model obj
        return BaselineSeq2Seq(parsed_model_args, self.pt_embedding, device=self.device, 
                               use_charlm=use_charlm, charlm_forward_file=charlm_forward_file, charlm_backward_file=charlm_backward_file)

    def load_model_from_checkpoint(self, checkpoint_load_path: str, device: str) -> BaselineSeq2Seq:
        """
        Loads a Seq2Seq model from its checkpoint state in `checkpoint_load_path`.

        Args:
            checkpoint_load_path (str): Path to the saved model file to load from.
            device (str): Which device to move the model components to
        """
        model = torch.load(checkpoint_load_path)
        # If the new model args specify to use coverage, then initialize the coverage if it doesn't already exist
        if self.model_args.get("coverage") and not model.coverage:  
            model.coverage = True 
            model.decoder.coverage = True
            model.decoder.coverage_vec = None
            model.decoder.attention.coverage = True
            model.decoder.attention.W_c = nn.Linear(1, model.decoder.decoder_hidden_dim, device=device)
        return model

    def compute_validation_loss(self, save_path: str, eval_file: str) -> float:
        """
        Computes the loss across the validation set for model selection across epochs

        Args:
            save_path (str): Path to the saved model checkpoint to evaluate
            eval_file (str): Path to the directory containing validation set files with examples
        """
        batch_size = self.model_args.get("batch_size", DEFAULT_BATCH_SIZE)
        device = default_device()
        val_set = Dataset(
                          data_root=eval_file,
                          batch_size=batch_size,
                          shuffle=True
                          )
        model = torch.load(save_path)
        model = model.to(device)
        model.eval()  # disable dropout

        # Load loss function
        self.criterion = nn.NLLLoss(reduction="none")
        self.criterion = self.criterion.to(device)
        running_loss = 0.0
        with torch.no_grad():
            for articles, summaries in tqdm(val_set, desc="Computing validation loss..."):

                if self.max_enc_steps is not None:  # truncate text
                    articles = [article[: self.max_enc_steps] for article in articles]
                if self.max_dec_steps is not None:  # truncate target
                    summaries = [summary[: self.max_dec_steps] for summary in summaries]
                    # if the truncated summary has no STOP token, add one
                    summaries = [summary + [STOP_TOKEN] if summary[-1] != STOP_TOKEN else summary for summary in summaries]

                output, attention_scores, coverage_vectors = model(articles, summaries, 0.0)  # (batch size, seq len, vocab size). Turn off teacher-forcing
                output = output.permute(0, 2, 1)   # (batch size, vocab size, seq len)

                target_indices = convert_text_to_token_ids(model.vocab_map, summaries, UNK_ID, self.max_dec_steps + 1 if self.max_dec_steps is not None else None).to(device)  # self.max_dec_steps + 1 because of STOP token

                # Compute losses (base loss)
                log_loss = self.criterion(output, target_indices)
                # coverage loss
                if model.coverage:
                    coverage_losses = torch.sum(torch.min(attention_scores, coverage_vectors), dim=-1)
                    combined_losses = log_loss + coverage_losses
                else:
                    combined_losses = log_loss 
                
                # backwards
                sequence_loss = combined_losses.mean(dim=1)
                batch_loss = sequence_loss.mean()

                running_loss += batch_loss.item()
        avg_loss = running_loss / len(val_set)
        return avg_loss
            

    def train(self, num_epochs: int, save_name: str, train_file: str, eval_file: str, checkpoint_load_path: str = None) -> None:
        """
        Trains a model on batches of texts

        Args:
            num_epochs (int): Number of training epochs 
            save_name (str): Path to store trained model
            eval_file (str): Path to the validation set file roots evaluating model checkpoints
            train_file (str): Path to training data root containing chunked files with tokenized text for each article + summary
            checkpoint_load_path (str): Path to model checkpoint to begin training from if provided.

        Returns:
            None (model with best validation loss will be saved to the save file)
        """
        model_chkpt_path = generate_checkpoint_path(save_name)  # adds 'ckpt' to model save path name. Used for eval
        best_loss = float('inf')
        device = default_device()
        # Load model in
        if checkpoint_load_path is not None and os.path.exists(checkpoint_load_path):  # load chkpt
            logger.info(f"Loading model checkpoint to start from: {checkpoint_load_path}")
            self.model = self.load_model_from_checkpoint(
                                                        checkpoint_load_path=checkpoint_load_path,
                                                        device=device,
                                                        )
        else:  # train a new model from scratch
            logger.info(f"Training model from scratch. Will be saved to {save_name}")
            self.model = self.build_model()

        self.model.to(device)
        logger.info(f"Model moved to device {device}.")

        # Get dataset 
        batch_size = self.model_args.get("batch_size", DEFAULT_BATCH_SIZE)
        dataset = Dataset(train_file, batch_size)
        PADDING_TOKEN_ID = self.model.vocab_map.get(PADDING_TOKEN)

        # Load optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        self.criterion = nn.NLLLoss(reduction="none", ignore_index=PADDING_TOKEN_ID)
        self.criterion = self.criterion.to(next(self.model.parameters()).device)

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            logger.info(f"Starting new epoch: {epoch + 1} / {num_epochs}")
            for articles, summaries in tqdm(dataset, desc="Training on examples..."):
                
                if self.max_enc_steps is not None:  # truncate text
                    articles = [article[: self.max_enc_steps] for article in articles]
                if self.max_dec_steps is not None:  # truncate target
                    summaries = [summary[: self.max_dec_steps] for summary in summaries]
                    # if the truncated summary has no STOP token, add one
                    summaries = [summary + [STOP_TOKEN] if summary[-1] != STOP_TOKEN else summary for summary in summaries]

                # Get model output
                self.optimizer.zero_grad()
                output, attention_scores, coverage_vectors = self.model(articles, summaries, 1.0)  # (batch size, seq len, vocab size)
                output = output.permute(0, 2, 1)   # (batch size, vocab size, seq len)

                target_indices = convert_text_to_token_ids(self.model.vocab_map, summaries, UNK_ID, self.max_dec_steps + 1 if self.max_dec_steps is not None else None).to(device)  # + 1 because of STOP token

                # Compute losses (base loss)
                log_loss = self.criterion(output, target_indices)
                # coverage loss
                if self.model.coverage:
                    coverage_losses = torch.sum(torch.min(attention_scores, coverage_vectors), dim=-1)

                    combined_losses = log_loss + coverage_losses
                else:
                    combined_losses = log_loss 
                
                # backwards
                sequence_loss = combined_losses.mean(dim=1)
                batch_loss = sequence_loss.mean()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)  # add gradient clipping at a max of 2.0 grad norm

                running_loss += batch_loss.item()
                self.optimizer.step()
            
            # Evaluate current model checkpoint
            epoch_loss = running_loss / len(dataset)
            logger.info(f"Epoch {epoch + 1} / {num_epochs}, Loss: {epoch_loss:.6f}")
            torch.save(self.model, model_chkpt_path)  

            val_set_loss = self.compute_validation_loss(
                                                        save_path=model_chkpt_path,
                                                        eval_file=eval_file 
                                                        )
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}] average validation loss: {val_set_loss}.")
            if val_set_loss < best_loss:
                best_loss = val_set_loss
                torch.save(self.model, save_name)
                logger.info(f"New best model saved to {save_name}! Val set loss: {best_loss}")


def parse_args():
    parser = argparse.ArgumentParser()
    # Model args
    parser.add_argument("--enc_hidden_dim", type=int, default=DEFAULT_ENCODER_HIDDEN_DIM, help="Size of encoder hidden states")
    parser.add_argument("--enc_num_layers", type=int, default=DEFAULT_ENCODER_NUM_LAYERS, help="Number of layers in the encoder LSTM")
    parser.add_argument("--dec_hidden_dim", type=int, default=DEFAULT_DECODER_HIDDEN_DIM, help="Size of decoder hidden state vector")
    parser.add_argument("--dec_num_layers", type=int, default=DEFAULT_DECODER_NUM_LAYERS, help="Number of layers in the decoder LSTM")
    parser.add_argument("--pgen", action="store_true", dest="pgen", default=False, help="Use pointergen probabilities to point to input text")
    parser.add_argument("--coverage", action="store_true", dest="coverage", default=False, help="Use coverage vectors during decoding stage")
    # Training args
    parser.add_argument("--checkpoint_load_path", type=str, default=None, help="If training from a checkpoint, the path to the checkpoint. Defaults to None.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for data processing")
    parser.add_argument("--save_name", type=str, default=DEFAULT_SAVE_NAME, help="Path to destination for final trained model.")
    parser.add_argument("--eval_path", type=str, default=DEFAULT_EVAL_ROOT, help="Path to the validation set root")
    parser.add_argument("--train_path", type=str, default=DEFAULT_TRAIN_ROOT, help="Path to the training data root")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--wordvec_pretrain_file", type=str, default=DEFAULT_WORDVEC_PRETRAIN_FILE, help="Path to pretrained word embeddings file")
    parser.add_argument("--charlm", action="store_true", dest="charlm", default=False, help="Use character language model embeddings.")
    parser.add_argument("--charlm_forward_file", type=str, default=os.path.join(os.path.dirname(__file__), "charlm_files", "1billion_forward.pt"), help="Path to forward charlm file")
    parser.add_argument("--charlm_backward_file", type=str, default=os.path.join(os.path.dirname(__file__), "charlm_files", "1billion_backwards.pt"), help="Path to backward charlm file")
    parser.add_argument("--max_enc_steps", type=int, default=None, help="Limit on article sizes (will be truncated)")
    parser.add_argument("--max_dec_steps", type=int, default=None, help="Limit on summary sizes (will be truncated)")
    return parser

def main():
    argparser = parse_args()
    args = argparser.parse_args()

    enc_hidden_dim = args.enc_hidden_dim
    enc_num_layers = args.enc_num_layers
    dec_hidden_dim = args.dec_hidden_dim
    dec_num_layers = args.dec_num_layers
    pgen = args.pgen
    coverage = args.pgen

    checkpoint_load_path = args.checkpoint_load_path
    batch_size = args.batch_size
    save_name = args.save_name
    eval_path = args.eval_path
    train_path = args.train_path
    num_epochs = args.num_epochs
    lr = args.lr
    wordvec_pretrain_file = args.wordvec_pretrain_file
    charlm_forward_file = args.charlm_forward_file
    charlm_backward_file = args.charlm_backward_file
    use_charlm = args.charlm
    max_enc_steps = args.max_enc_steps
    max_dec_steps = args.max_dec_steps

    if checkpoint_load_path is not None:
        if not os.path.exists(checkpoint_load_path):
            no_chkpt_msg = f"Could not find checkpoint loading file: {checkpoint_load_path}"
            logger.error(no_chkpt_msg)
            raise FileNotFoundError(no_chkpt_msg)
    if not os.path.exists(eval_path):
        no_eval_file_msg = f"Could not find provided eval dir: {eval_path}"
        logger.error(no_eval_file_msg)
        raise FileNotFoundError(no_eval_file_msg)
    if not os.path.exists(train_path):
        no_train_file_msg = f"Could not find provided train dir: {train_path}"
        logger.error(no_train_file_msg)
        raise FileNotFoundError(no_train_file_msg)
    if not os.path.exists(wordvec_pretrain_file):
        no_wordvec_file_msg = f"Could not find provided wordvec pretrain file {wordvec_pretrain_file}"
        logger.error(no_wordvec_file_msg)
        raise FileNotFoundError(no_wordvec_file_msg)
    if use_charlm:
        if not os.path.exists(charlm_forward_file):
            no_charlm_forward_file_msg = f"Could not find provided charlm forward file {charlm_forward_file}"
            logger.error(no_charlm_forward_file_msg)
            raise FileNotFoundError(no_charlm_forward_file_msg)
        if not os.path.exists(charlm_backward_file):
            no_charlm_backward_file_msg = f"Could not find provided charlm backward file {charlm_backward_file}"
            logger.error(no_charlm_backward_file_msg)
            raise FileNotFoundError(no_charlm_backward_file_msg)
    
    args = vars(args)
    logger.info("Using the following args for training:")
    for arg, val in args.items():
        logger.info(f"{arg}: {val}")

    trainer = SummarizationTrainer(
        model_args=args,
        embedding_file=wordvec_pretrain_file,
        lr=lr
    )
    trainer.train(
        num_epochs=num_epochs,
        save_name=save_name,
        train_file=train_path,
        eval_file=eval_path,
        checkpoint_load_path=checkpoint_load_path
    )


if __name__ == "__main__":
    main()