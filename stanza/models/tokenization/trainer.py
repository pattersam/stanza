import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim

from stanza.models.common import utils
from stanza.models.common.trainer import Trainer as BaseTrainer
from stanza.models.tokenization.utils import create_dictionary

from .model import Tokenizer
from .vocab import Vocab

logger = logging.getLogger('stanza')

class Trainer(BaseTrainer):
    def __init__(self, args=None, vocab=None, lexicon=None, dictionary=None, model_file=None, device=None, foundation_cache=None):
        # TODO: make a test of the training w/ and w/o charlm
        # TODO: update the run_tokenizer script to easily support charlm
        #   - needs to have the model filename built with charlm
        # TODO: build the resources with the forward charlm
        if model_file is not None:
            # load everything from file
            self.load(model_file, args, foundation_cache)
        else:
            # build model from scratch
            self.args = args
            self.vocab = vocab
            self.lexicon = lexicon
            self.dictionary = dictionary
            self.model = Tokenizer(self.args, self.args['vocab_size'], self.args['emb_dim'], self.args['hidden_dim'], dropout=self.args['dropout'], feat_dropout=self.args['feat_dropout'])
        self.model = self.model.to(device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
        self.optimizer = utils.get_optimizer("adam", self.model, lr=self.args['lr0'], betas=(.9, .9), weight_decay=self.args['weight_decay'])
        self.feat_funcs = self.args.get('feat_funcs', None)
        self.lang = self.args['lang'] # language determines how token normalization is done

    def update(self, inputs):
        self.model.train()
        units, labels, features, raw_units = inputs

        device = next(self.model.parameters()).device
        units = units.to(device)
        labels = labels.to(device)
        features = features.to(device)

        pred = self.model(units, features, raw_units)

        self.optimizer.zero_grad()
        classes = pred.size(2)
        loss = self.criterion(pred.view(-1, classes), labels.view(-1))

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()

        return loss.item()

    def predict(self, inputs):
        self.model.eval()
        units, _, features, raw_units = inputs

        device = next(self.model.parameters()).device
        units = units.to(device)
        features = features.to(device)

        pred = self.model(units, features, raw_units)

        return pred.data.cpu().numpy()

    def save(self, filename, skip_modules=True):
        model_state = None
        if self.model is not None:
            model_state = self.model.state_dict()
            # skip saving modules like the pretrained charlm
            if skip_modules:
                skipped = [k for k in model_state.keys() if k.split('.')[0] in self.model.unsaved_modules]
                for k in skipped:
                    del model_state[k]

        params = {
            'model': model_state,
            'vocab': self.vocab.state_dict(),
            'lexicon': self.lexicon,
            'config': self.args
        }
        try:
            torch.save(params, filename, _use_new_zipfile_serialization=False)
            logger.info("Model saved to {}".format(filename))
        except BaseException:
            logger.warning("Saving failed... continuing anyway.")

    def load(self, filename, args, foundation_cache):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            logger.error("Cannot load model from {}".format(filename))
            raise
        self.args = checkpoint['config']
        if args is not None and args.get('charlm_forward_file', None) is not None:
            self.args['charlm_forward_file'] = args['charlm_forward_file']
        if self.args.get('use_mwt', None) is None:
            # Default to True as many currently saved models
            # were built with mwt layers
            self.args['use_mwt'] = True
        self.model = Tokenizer(self.args, self.args['vocab_size'], self.args['emb_dim'], self.args['hidden_dim'], dropout=self.args['dropout'], feat_dropout=self.args['feat_dropout'])
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.vocab = Vocab.load_state_dict(checkpoint['vocab'])
        self.lexicon = checkpoint['lexicon']

        if self.lexicon is not None:
            self.dictionary = create_dictionary(self.lexicon)
        else:
            self.dictionary = None
