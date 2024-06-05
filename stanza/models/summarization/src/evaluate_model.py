"""
Evaluates a trained abstractive summarization Seq2Seq model
"""
import argparse
import sys
import os 
import torch

ROOT = '/Users/alexshan/Desktop/stanza'
sys.path.append(ROOT)

import evaluate
import logging
from typing import List, Tuple, Mapping
from stanza.models.summarization.src.decode import BeamSearchDecoder
from stanza.models.summarization.src.model import BaselineSeq2Seq
from stanza.models.common.vocab import BaseVocab
from stanza.models.common.utils import default_device

logger = logging.getLogger('stanza.summarization') 
logger.propagate = False

# Check if the logger has handlers already configured
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def evaluate_predictions_rouge(generated_summaries: List[str], reference_summaries: List[str]):
    """
    Computes the ROUGE scores between a list of generated summaries and their corresponding reference summaries.

    Args:
        generated_summaries (List[str]): A list of summaries generated by a trained summarization model
        reference_summaries (List[str]): A list of summaries that are gold standard answers from test set.
    """
    rouge = evaluate.load('rouge')
    results = rouge.compute(
                            predictions=generated_summaries,
                            references=reference_summaries,
                            use_aggregator=True
                            )
    
    for k, v in results.items():
        logger.info(f"{k}: {v:.4f}")
    return results


def evaluate_model_rouge(model_path: str, articles: List[List[str]], summaries: List[List[str]], logger: logging.Logger = None,
                         max_enc_steps: int = None, max_dec_steps: int = None):

    """
    Evaluates a model on a set of articles and summaries by generating its own summaries from the articles
    and computing the ROUGE scores between its generated summaries and the reference summaries provided.

    Args: 
        model_path (str): Path to the trained and saved model checkpoint
        articles (List[List[str]]): A list of articles where each word is tokenized as a separate List entry
        summaries (List[List[str]]): A list of corresponding summaries where each word is tokenized as a separate List entry
        logger (Logger, optional): Logger object used to print supplementary info during evaluation.
        max_enc_steps (int, optional): Limit on the number of tokens per article. Defaults to no limit.
        max_dec_stpes (int, optional): Limit on the number of tokens per summary. Defaults to no limit.
    """
    device = default_device()
    trained_model = torch.load(model_path)
    trained_model = trained_model.to(device)
    trained_model.eval()
    logger.info(f"Successfully loaded model at {model_path} for evaluation.")

    decoder = BeamSearchDecoder(trained_model)

    generated_summaries = decoder.decode_examples(
                                                 examples=articles,
                                                 beam_size=4,
                                                 max_dec_steps=max_dec_steps,
                                                 min_dec_steps=10,  # TODO make this toggle, reference paper
                                                 max_enc_steps=max_enc_steps,
                                                 verbose=False,       
                                                 )
    generated_summaries = [" ".join(summary) for summary in generated_summaries]
    summaries = [" ".join(summary) for summary in summaries]
    
    results = evaluate_predictions_rouge(generated_summaries, summaries)
    return results


def evaluate_from_path(model_path: str, eval_path: str, logger: logging.Logger = None, 
                       max_enc_steps: int = None, max_dec_steps: int = None):

    """
    Evaluates a trained summarization model given a path to the directory containing 
    chunked files used for evaluation. 

    Args:
        model_path (str): Path to trained and saved model checkpoint.
        eval_path (str): Path to root of the eval set directory containing chunked files with examples.
        logger (Logger, optional): Logger object used to print supplementary info during evaluation.
        max_enc_steps (int, optional): Limit on the number of tokens per article. Defaults to no limit.
        max_dec_stpes (int, optional): Limit on the number of tokens per summary. Defaults to no limit.
    """
    
    # Get data
    articles, summaries = [], []
    chunked_files = os.listdir(eval_path)
    data_paths = [os.path.join(eval_path, chunked) for chunked in chunked_files]

    for path in data_paths:  
        with open(path, "r+", encoding='utf-8') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):  # iterate through lines in increments of two, getting article + summary
                article, summary = lines[i].strip("\n"), lines[i + 1].strip("\n")
                tokenized_article, tokenized_summary = article.split(" "), summary.split(" ")
                articles.append(tokenized_article)
                summaries.append(tokenized_summary)

    logger.info(f"Successfully loaded dataset for evaluation.")
    
    if max_enc_steps is not None:  # truncate input article
        articles = [article[: max_enc_steps] for article in articles] 
    if max_dec_steps is not None:  # truncate summaries
        summaries = [summary[: max_dec_steps] for summary in articles]

    results = evaluate_model_rouge(
                   model_path, 
                   articles, 
                   summaries, 
                   logger,
                   max_enc_steps=max_enc_steps,
                   max_dec_steps=max_dec_steps,
                   )
    return results


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="", help="Path to trained model file.")
    parser.add_argument("--eval_path", type=str, default="", help="Path to directory containing chunked test files.")
    parser.add_argument("--max_enc_steps", type=int, default=None, help="Limit on the number of tokens per article")
    parser.add_argument("--max_dec_steps", type=int, default=None, help="Limit on the number of tokens per summary")
    
    args = parser.parse_args()

    model_path = args.model_path
    eval_path = args.eval_path
    max_enc_steps = args.max_enc_steps
    max_dec_steps = args.max_dec_steps

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Expected to find model in {model_path}.")
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Expected to find directory {eval_path}.")
    
    logger.info(f"Using the following args for evaluating model: ")
    args = vars(args)
    for k, v in args.items():
        logger.info(f"{k}: {v}")
    
    evaluate_from_path(
        model_path,
        eval_path,
        logger=logger,
        max_enc_steps=max_enc_steps,
        max_dec_steps=max_dec_steps, 
    )

if __name__ == "__main__":
    main()