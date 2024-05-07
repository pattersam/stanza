import torch
import torch.nn as nn
import os
import logging
import math 
import torch.nn.functional as F


from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from stanza.models.common.char_model import CharacterModel, CharacterLanguageModel
from stanza.models.common.foundation_cache import load_pretrain

from stanza.models.summarization.constants import * 
from typing import List, Tuple
from copy import deepcopy

logger = logging.getLogger('stanza.lemmaclassifier')
torch.set_printoptions(threshold=100, edgeitems=5, linewidth=100)


"""
Overall structure

Embedding layer
Bi-LSTM Encoder
Uni-LSTM Decoder

"""


class BaselineEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(BaselineEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, text):
        outputs, (hidden, cell) = self.lstm(text)
        unpacked_lstm_outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) 
        cell = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)


        hidden_linearized = self.hidden_out(hidden)   # apply linear layer to reduce dimensionality -> initial decoder state
        cell_linearized = self.hidden_out(cell)
        return unpacked_lstm_outputs, hidden, cell, hidden_linearized, cell_linearized


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, coverage=False):
        super(BahdanauAttention, self).__init__()
        self.coverage = coverage
        self.W_h = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.W_s = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)
        self.b_attn = nn.Parameter(torch.rand(decoder_hidden_dim))

        self.W_c = None
        if self.coverage:
            # self.W_c = nn.Linear(17, decoder_hidden_dim)  # replace 17 with seqlen, or maybe this should be max dec steps
            # self.W_c = nn.Conv1d(in_channels=1, out_channels=decoder_hidden_dim, kernel_size=1, bias=False)
            self.W_c = nn.Linear(1, decoder_hidden_dim)

    
    def forward(self, encoder_outputs, decoder_hidden, coverage_vec=None):
        seq_len = encoder_outputs.shape[1]
        batch_size = encoder_outputs.shape[0]

        # Repeat decoder hidden state seq_len times
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Compute energy scores with bias term
        # print(f"Coverage = {self.coverage}, {coverage_vec.shape}")

        """
        For a single batch example:

        the energy vector at timestep t, e^t, has shape (sequence length).
        the i-th element of the energy vector, e_i^t, is a scalar computed with c_i^t, also a scalar.
        this implies that the coverage vector at timestep t, c^t, has shape (sequence length)

        Therefore, we want to design the w_c term to have size (hidden dim), such that multiplying
        the two gives us shape (sequence length, hidden dim) which might require unsqueezing the w_c
        vector along the sequence length direction so that at the i-th hidden state, we have the same w_c.
        So what we really want is a tensor of shape (hidden dim, sequence length) for the W_c

        We want out (batch size, sequence length, hidden dim). So our input should be shape
        (batch size, sequence length, sequence length). We need to copy the dim of the coverage vec. 

        
        """
        energy = None
        if self.coverage:
            print("Entered into here")
            print(f"Coverage vec shape {coverage_vec.shape}, Decoder hidden shape {decoder_hidden.shape}, {self.b_attn.shape}")
            # print(f"{self.W_c(coverage_vec.unsqueeze(1)).squeeze(1).transpose(1, 2).shape},{self.W_h(encoder_outputs).shape}, {self.W_s(decoder_hidden).shape}")
            # energy = torch.tanh(self.W_h(encoder_outputs) + self.W_s(decoder_hidden) + self.W_c(coverage_vec.unsqueeze(1)).squeeze(1).transpose(1, 2) + self.b_attn)
            print(f"New coverage vec shape {coverage_vec.unsqueeze(-1).shape}, {coverage_vec.unsqueeze(-1)}")
            energy = torch.tanh(self.W_h(encoder_outputs) + self.W_s(decoder_hidden) + self.W_c(coverage_vec.unsqueeze(-1)) + self.b_attn)
        else:
            energy = torch.tanh(self.W_h(encoder_outputs) + self.W_s(decoder_hidden) + self.b_attn)

        print(f"Energy scores shape: {energy.shape}")

        attention = self.v(energy).squeeze(2)

        print(f"Attn shape {attention.shape}")

        # Generate mask: valid tokens have a hidden state different from zero
        mask = (encoder_outputs.abs().sum(dim=2) > 0).float()

        # Apply the mask by setting invalid positions to -inf
        attention = attention.masked_fill(mask == 0, float('-inf'))
        attention =  F.softmax(attention, dim=1)

        if self.coverage and coverage_vec is not None:
            coverage_vec = coverage_vec + attention
        return attention, coverage_vec


class BaselineDecoder(nn.Module):

    def __init__(self, output_dim, encoder_hidden_dim, decoder_hidden_dim, emb_dim, num_layers=1, use_pgen=False, use_coverage=False):
        super(BaselineDecoder, self).__init__()

        self.output_dim = output_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.encoder_hidden_dim = encoder_hidden_dim

        self.lstm = nn.LSTM(emb_dim, decoder_hidden_dim, num_layers=num_layers, batch_first=True)   
        self.attention = BahdanauAttention(encoder_hidden_dim, decoder_hidden_dim, coverage=use_coverage)
                
        self.pgen = use_pgen
        self.coverage = use_coverage

        if self.pgen:
            self.p_gen_linear = nn.Linear(encoder_hidden_dim * 2 + emb_dim + decoder_hidden_dim, 1) 

        if self.coverage:
            self.coverage_vec = None
         
        # Two linear layers as per equation (4) in the paper
        self.V = nn.Linear(encoder_hidden_dim * 2 + decoder_hidden_dim, decoder_hidden_dim)
        self.V_prime = nn.Linear(decoder_hidden_dim, output_dim)
        
        self.softmax = nn.Softmax(dim=1)  # Softmax layer for the final output

    def forward(self, input, hidden, cell, encoder_outputs, src=None):

        """
        input : (batch size, emb dim)
        hidden : (batch size, decoder hidden dim)
        cell : (batch size, decoder hidden dim)
        encoder_outputs (batch size, seq len, encoder hidden dim)
        """
        
        batch_size = input.shape[0]
        sequence_length = encoder_outputs.shape[1]  
        print(f"Using seqlen {sequence_length}")
        if self.coverage and self.coverage_vec is None:
            self.coverage_vec = torch.zeros(batch_size, sequence_length)
        
        # Attention is computed using the decoder's current hidden state 'hidden' and all the encoder outputs
        attention_weights, coverage_vec = self.attention(encoder_outputs, hidden, coverage_vec=self.coverage_vec if self.coverage else None)
        if self.coverage:
            self.coverage_vec = coverage_vec
            print(f"Coverage vector: {self.coverage_vec}, {self.coverage_vec.shape}")

        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context_vector = context_vector.squeeze(1)  # (batch size, 2 * encoder hidden dim)

        # Here 'hidden' is the current state of the decoder, also known as decoder state
        # 'lstm_output' is the output of the LSTM at the current step, which can sometimes be different from 'hidden'
        # especially when using LSTM cells, since 'lstm_output' may be the output from the top layer of a multi-layer LSTM
        input = input.unsqueeze(1)  # (batch size, 1, embedding dim)
        hidden = hidden.unsqueeze(1).transpose(0, 1)
        cell = cell.unsqueeze(1).transpose(0, 1)

        lstm_output, (hidden, cell) = self.lstm(input, (hidden, cell))

        hidden, cell = hidden.transpose(0, 1), cell.transpose(0, 1)
        hidden, cell = hidden.squeeze(1), cell.squeeze(1)  # 'hidden' is now the updated decoder state after processing the current input token
        # This 'hidden' will be used in the next time step's attention computation
        # hidden & cell shape (batch size, decoder hidden dim)

        p_gen = None
        if self.pgen:
            p_gen_input = torch.cat((context_vector, hidden, input.squeeze(1)), dim=1)  # (batch size, 2 * encoder hidden dim + decoder hidden dim + emb dim)
            p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input))  # (batch size, 1)

        # The paper states that the decoder state (hidden) and the context vector are concatenated
        # before being passed through linear layers to predict the next token.
        concatenated = torch.cat((hidden, context_vector), dim=1)

        output = self.V(concatenated) 
        output = self.V_prime(output) 

        p_vocab = self.softmax(output)

        return p_vocab, hidden, cell, attention_weights, p_gen


class BaselineSeq2Seq(nn.Module):
    """
    
    """
    def __init__(self, model_args, pt_embedding):
        super(BaselineSeq2Seq, self).__init__()
        self.model_args = model_args
        self.input_size = 0
        self.batch_size = self.model_args.get("batch_size", DEFAULT_BATCH_SIZE)
        
        emb_matrix = pt_embedding.emb   # have to load this in through file by using 'load_pretrain' helper
        self.vocab_size = emb_matrix.shape[0]
        self.embedding_dim = emb_matrix.shape[1]

        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=False)   # freeze False because 'See et. al.' updates embeddings
        self.vocab_map = {word.replace('\xa0', ' '): i for i, word in enumerate(pt_embedding.vocab)}

        # extended vocab 
        self.ext_vocab_map = deepcopy(self.vocab_map)
        self.max_oov_words = 0

        self.input_size += self.embedding_dim

        encoder_hidden_dim = self.model_args.get("encoder_hidden_dim", DEFAULT_ENCODER_HIDDEN_DIM)
        encoder_num_layers = self.model_args.get("encoder_num_layers", DEFAULT_ENCODER_NUM_LAYERS)
        self.encoder = BaselineEncoder(self.input_size, encoder_hidden_dim, num_layers=encoder_num_layers)

        decoder_hidden_dim = self.model_args.get("decoder_hidden_dim", encoder_hidden_dim)   # default value should be same hidden dim as encoder
        decoder_num_layers = self.model_args.get("decoder_num_layers", encoder_num_layers)
        self.pgen = self.model_args.get("pgen", False)
        self.coverage = self.model_args.get("coverage", False)

        # TODO Remove this after testing.
        if self.coverage:
            print(f"Using coverage!")

        self.decoder = BaselineDecoder(self.vocab_size, encoder_hidden_dim, decoder_hidden_dim, self.embedding_dim, decoder_num_layers, self.pgen, self.coverage)
    

    def extract_word_embeddings(self, text: List[List[str]]):
        """
        Extracts the word embeddings over the input articles in 'text'.

        text (List[List[str]]): Tokenized articles of text

        Returns a tensor of the padded embeddings over the inputs. Also returns the input lengths.
        """
        token_ids, input_lengths = [], []
        for article in text:
            article_token_ids = torch.tensor([self.vocab_map.get(word.lower(), UNK_ID) for word in article])
            token_ids.append(article_token_ids)
            input_lengths.append(len(article_token_ids))
        padded_inputs = pad_sequence(token_ids, batch_first=True)
        embedded = self.embedding(padded_inputs)
        return embedded, input_lengths
    
    
    def build_extended_vocab_map(self, src: List[List[str]]):
        """
        constructs the extended vocabulary map between source document words and their indices
        
        for each document, the extended vocabulary is the union of the src document words and the existing vocab

        returns size (batch size, vocab size + num oov words)


        Instead of the index tensor being all zeros, which is an issue because for index tensors that don't get
        filled all the way, you get zero, which can be interpreted as zero index later. 

        As long as the attn mask is computed correctly, it should be okay for the index tensor to be zeroes.
        If the attn is computed, then out of sequence words get 0 attention.
        """
        batch_size = len(src)
        max_seq_len = max([len(doc) for doc in src])
        max_oov_words = 0

        index_tensor = torch.zeros(batch_size, max_seq_len, dtype=torch.int64)
        for batch_idx, document in enumerate(src):
            num_oov_words = 0
            doc_indexes = torch.zeros(max_seq_len)
            for i, word in enumerate(document):
                vocab_idx = self.ext_vocab_map.get(word.lower()) 

                if vocab_idx is None:
                    # If we cannot find the current word, we add it to the extended vocab
                    self.ext_vocab_map[word.lower()] = len(self.ext_vocab_map)  # new slot
                    num_oov_words += 1

                revised_idx = self.ext_vocab_map.get(word.lower())
                if revised_idx is None:
                    raise ValueError(f"Error building extended vocab map, word: {word}")
                doc_indexes[i] = revised_idx

            index_tensor[batch_idx] = doc_indexes
            max_oov_words = max(max_oov_words, num_oov_words)

        return index_tensor, max_oov_words


    def forward(self, text, target, teacher_forcing_ratio=0.5):
        """
        text: (List[List[str]])
        target: (List[List[str]])

        """
        batch_size = min(len(text), self.batch_size) # TODO later fix 

        index_tensor, max_oov_words = self.build_extended_vocab_map(text)  # (batch size, seq len)
        self.max_oov_words = max_oov_words

        # Get embeddings over the input text
        embedded, input_lengths = self.extract_word_embeddings(text)

        # Get embeddings over the target text
        target_embeddings, target_lengths = self.extract_word_embeddings(target)
        target_len = target_embeddings.shape[1]   # TODO : Ask John how batch processing works with this. should this actually just be a uniform hyperparam like max_dec_steps? If so, how to do padding?

        # Tensor to store decoder outputs
        effective_vocab_size = self.vocab_size + self.max_oov_words if self.pgen else self.vocab_size
        outputs = torch.zeros(batch_size, target_len, effective_vocab_size)

        packed_input_seqs = pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        # packed_target_seqs = pack_padded_sequence(target_embeddings, target_lengths, batch_first=True, enforce_sorted=False)

        # Embeddings fed into Encoder LSTM
        # Get the hidden states h_i from the encoder
        # Take the encoder's hidden state after being passed through the linear layer into the decoder for the initial state
        encoder_outputs, encoder_hidden, encoder_cell, decoder_init_state, decoder_init_cell = self.encoder(packed_input_seqs)

        # For each decoder time step, the decoder receives the word embedding of the previous word 
        input = target_embeddings[:, 0, :] # TODO make sure the data uses <sos> at the beginning of sentences

        hidden = decoder_init_state   # the initial decoder hidden state is the linearized hidden state of the encoder
        cell = decoder_init_cell   # similar for the cell ^

        for t in range(target_len):
            p_vocab, hidden, cell, attn_weights, pgen = self.decoder(input, hidden, cell, encoder_outputs)

            # if no pgen, then our final dist is p_vocab. otherwise, calculate the final distribution
            if self.pgen:   
                p_vocab_scaled = pgen * p_vocab   # (batch size, vocab size)
                attn_dist_scaled = (1 - pgen) * attn_weights   # (batch size, seq len)


                """
                For each word in the sequence, we need to know if it is in the vocabulary or not. 
                If the word is is in the vocab, then its value in the extended vocabulary distribution 
                is p_gen * P_vocab(w) + (1 - p_gen) * sum_i a_i^t

                If the word is not in the vocab, then it receives a new index in the extended vocab and 
                its value is (1 - p_gen) * sum_i a_i^t

                The extended vocab will be shape (batch size, extended vocab size)
                So for the batches, it will be (batch size, max extended vocab size)
                We start with zeroes, copy over the scaled P_vocab distribution, and then add where appropriate?

                So, we need a tensor of shape (batch size, seq len) that gives each word's index in the vocab.
                Once we have the word's index in the vocab, we know which index to add the summation term to.
                For new words, we create a new index. We also need an extended vocab map.
                """

                # at this point, assume that we have a tensor where for each batch, the i-th index of the tensor
                # contains the index of the sequence token in the extended vocab

                # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
                extended_vocab_size = self.vocab_size + self.max_oov_words 
                extended_vocab_dist = torch.zeros(batch_size, extended_vocab_size)
                extended_vocab_dist[:, :self.vocab_size] = p_vocab_scaled  # add the existing distribution to the extended vocab

                final_vocab_dist = extended_vocab_dist.scatter_add_(dim=1, index=index_tensor, src=attn_dist_scaled)
                p_vocab = final_vocab_dist

            # Place predictions in a tensor holding predictions for each token
            outputs[:, t, :] = p_vocab 

            # Decide whether to use teacher forcing or not
            teacher_force = torch.rand(1) < teacher_forcing_ratio

            # Get the highest predicted token from our predictions
            top1 = torch.argmax(p_vocab, dim=1)

            # If teacher forcing, use actual next token as next input. If not, use the predicted token.
            input = target_embeddings[:, t, :] if teacher_force else self.embedding(top1)  # TODO: Bug where if we select top1 to be an OOV word, then we need to have a valid embedding for the word
            # If you get an index out of range in self error, it's because of the bug. Just rerun until you don't get that error.
            
            print(f"Executed decoder timestep {t}")

        return outputs
