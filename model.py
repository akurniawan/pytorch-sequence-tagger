import torch.nn as nn

from torchcrf import CRF
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence)

from embedding import WordCharCNNEmbedding, WordCharLSTMEmbedding


class NERTagger(nn.Module):
    def __init__(self,
                 embedding,
                 nemb,
                 nhid,
                 nlayers,
                 drop,
                 ntags,
                 batch_first=True):
        super(NERTagger, self).__init__()
        self.embedding = embedding
        self.tagger_rnn = nn.LSTM(
            input_size=nemb,
            hidden_size=nhid,
            num_layers=nlayers,
            dropout=drop,
            bidirectional=True)
        self._init_rnn_weights()

        self.projection = nn.Sequential(
            nn.Linear(in_features=nhid * 2, out_features=ntags))
        self._init_linear_weights_and_bias()

        self.crf_tagger = CRF(ntags)
        self._batch_first = batch_first

    def _init_rnn_weights(self):
        """Initialize the weight of rnn using xavier and constant
        with value of one for bias
        """

        for idx in range(len(self.tagger_rnn.all_weights[0])):
            dim = self.tagger_rnn.all_weights[0][idx].size()
            if len(dim) < 2:
                nn.init.constant_(self.tagger_rnn.all_weights[0][idx], 1)
            elif len(dim) == 2:
                nn.init.xavier_uniform_(self.tagger_rnn.all_weights[0][idx])

    def _init_linear_weights_and_bias(self):
        """Initialize the weight of linear layer using xavier and constant
        with value of one for bias
        """
        # Init linear weights
        nn.init.xavier_uniform_(self.projection[0].weight)
        # Init bias weights
        nn.init.constant_(self.projection[0].bias, 1)

    def _rnn_forward(self, x, seq_len):
        packed_sequence = pack_padded_sequence(
            x, seq_len, batch_first=self._batch_first)
        out, _ = self.tagger_rnn(packed_sequence)
        out, lengths = pad_packed_sequence(out, batch_first=self._batch_first)
        projection = self.projection(out)

        return projection

    def forward(self, x, x_word, seq_len, y):
        embed = self.embedding(x, x_word)
        projection = self._rnn_forward(embed, seq_len)
        llikelihood = self.crf_tagger(projection, y)

        return -llikelihood

    def decode(self, x, x_word, seq_len):
        embed = self.embedding(x, x_word)
        projection = self._rnn_forward(embed, seq_len)
        result = self.crf_tagger.decode(projection)

        return result


def get_model_fn(embedding_config, tagger_config):
    def model_fn(sentence_field, char_sentence_field, tags_field):
        if embedding_config["embedding_type"] == "cnn":
            embedding = WordCharCNNEmbedding(
                word_num_embedding=len(sentence_field.vocab),
                word_embedding_dim=embedding_config["word_embedding_size"],
                word_padding_idx=sentence_field.vocab.stoi["<pad>"],
                char_num_embedding=len(char_sentence_field.vocab),
                char_embedding_dim=embedding_config["char_embedding_size"],
                char_padding_idx=char_sentence_field.vocab.stoi["<pad>"],
                dropout=embedding_config["embedding_dropout"],
                kernel_size=embedding_config["kernel_size"],
                out_channels=embedding_config["output_size"],
                pretrained_word_embedding=sentence_field.vocab.vectors)
        elif embedding_config["embedding_type"] == "lstm":
            embedding = WordCharLSTMEmbedding(
                word_num_embedding=len(sentence_field.vocab),
                word_embedding_dim=embedding_config["word_embedding_size"],
                word_padding_idx=sentence_field.vocab.stoi["<pad>"],
                char_num_embedding=len(char_sentence_field.vocab),
                char_embedding_dim=embedding_config["char_embedding_size"],
                char_padding_idx=char_sentence_field.vocab.stoi["<pad>"],
                dropout=embedding_config["embedding_dropout"],
                char_lstm_hidden_size=embedding_config["output_size"],
                char_lstm_layers=embedding_config["char_lstm_layers"],
                char_lstm_dropout=embedding_config["char_lstm_dropout"],
                pretrained_word_embedding=sentence_field.vocab.vectors)
        tagger = NERTagger(
            embedding=embedding,
            nemb=embedding_config["output_size"] +
            embedding_config["word_embedding_size"],
            nhid=tagger_config["hidden_size"],
            nlayers=tagger_config["layer_size"],
            drop=tagger_config["rnn_dropout"],
            ntags=len(tags_field.vocab))
        return tagger

    return model_fn
