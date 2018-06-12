import torch
import torch.nn as nn
import torch.nn.functional as F


class WordCharCNNEmbedding(nn.Module):
    """Combination between character and word embedding as the
    features for the tagger. The character embedding is built
    upon CNN and pooling layer.
    """

    def __init__(self,
                 padding_size=2,
                 word_num_embedding=100,
                 word_embedding_dim=300,
                 word_padding_idx=1,
                 char_num_embedding=30,
                 char_embedding_dim=30,
                 char_padding_idx=1,
                 dropout=0.5,
                 kernel_size=3,
                 out_channels=30,
                 pretrained_word_embedding=None):
        super(WordCharCNNEmbedding, self).__init__()
        self.char_embedding = nn.Embedding(
            char_num_embedding, char_embedding_dim, char_padding_idx)
        self._init_char_embedding(char_padding_idx)
        self.conv_embedding = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=(kernel_size, char_embedding_dim)))
        self.word_embedding = nn.Embedding(
            word_num_embedding, word_embedding_dim, word_padding_idx)
        if isinstance(pretrained_word_embedding, torch.Tensor):
            self.word_embedding.weight.data.copy_(pretrained_word_embedding)
            # Freeze the embedding layer when using pretrained word
            # embedding
            self.word_embedding.weight.requires_grad = False

    def _init_char_embedding(self, padding_idx):
        """Initialize the weight of character embedding with xavier
        and reinitalize the padding vectors to zero
        """

        nn.init.xavier_normal_(self.char_embedding.weight)
        # Reinitialize vectors at padding_idx to have 0 value
        self.char_embedding.weight.data[padding_idx].uniform_(0, 0)

    def forward(self, X, X_word):
        word_size = X.size(1)
        char_embeddings = []
        # We have X in dimension of [batch, words, chars]. To use
        # batch calculation we need to loop over all words and
        # calculate the embedding
        for i in range(word_size):
            # Convert the embedding size from [batch, chars]
            # into [batch, 1, chars]. 1 is our channel for
            # convolution layer later
            x = X[:, i, :].unsqueeze(1)
            # Apply embedding for every characters on batch.
            # The dimension now will be [batch, 1, chars, emb]
            char_embedding = self.char_embedding(x)
            # Apply char embedding with dropout and convolution
            # layers so the dim now will be [batch, conv_size, new_height, 1]
            char_embedding = self.conv_embedding(char_embedding)
            # Remove the last dimension with size 1
            char_embedding = char_embedding.squeeze(-1)
            # Apply pooling layer so the new dim will be [batch, conv_size, 1]
            char_embedding = F.max_pool2d(
                char_embedding,
                kernel_size=(1, char_embedding.size(2)),
                stride=1)
            # Transpose it before we put it into array for later concatenation
            char_embeddings.append(char_embedding.transpose(1, 2))

        # Concatenate the whole char embeddings
        final_char_embedding = torch.cat(char_embeddings, dim=1)
        word_embedding = self.word_embedding(X_word)

        # Combine both character and word embeddings
        result = torch.cat([final_char_embedding, word_embedding], 2)
        return result


class WordCharLSTMEmbedding(nn.Module):
    """Combination between character and word embedding as the
    features for the tagger. The character embedding is built uplon
    LSTM layer.
    """

    def __init__(self,
                 padding_size=2,
                 word_num_embedding=100,
                 word_embedding_dim=300,
                 word_padding_idx=1,
                 char_num_embedding=30,
                 char_embedding_dim=30,
                 char_padding_idx=1,
                 dropout=0.5,
                 char_lstm_hidden_size=50,
                 char_lstm_layers=1,
                 char_lstm_dropout=0.5,
                 pretrained_word_embedding=None):
        super(WordCharLSTMEmbedding, self).__init__()

        self._char_lstm_hidden_size = char_lstm_hidden_size
        self.char_embedding = nn.Embedding(
            char_num_embedding, char_embedding_dim, char_padding_idx)
        self._init_char_embedding(char_padding_idx)

        self.char_lstm_embedding = nn.LSTM(
            input_size=char_embedding_dim,
            hidden_size=char_lstm_hidden_size,
            num_layers=char_lstm_layers,
            dropout=char_lstm_dropout,
            batch_first=False,
            bidirectional=True)
        self._init_rnn_weights()
        self.char_linear_embedding = nn.Linear(
            in_features=2 * char_lstm_hidden_size,
            out_features=char_lstm_hidden_size)
        self._init_linear_weights_and_bias()
        self.word_embedding = nn.Embedding(
            word_num_embedding, word_embedding_dim, word_padding_idx)
        if isinstance(pretrained_word_embedding, torch.Tensor):
            self.word_embedding.weight.data.copy_(pretrained_word_embedding)
            # Freeze the embedding layer when using pretrained word
            # embedding
            self.word_embedding.weight.requires_grad = False

    def _init_char_embedding(self, padding_idx):
        """Initialize the weight of character embedding with xavier
        and reinitalize the padding vectors to zero
        """

        nn.init.xavier_normal_(self.char_embedding.weight)
        # Reinitialize vectors at padding_idx to have 0 value
        self.char_embedding.weight.data[padding_idx].uniform_(0, 0)

    def _init_rnn_weights(self):
        """Initialize the weight of rnn using xavier and constant
        with value of one for bias
        """

        for idx in range(len(self.char_lstm_embedding.all_weights[0])):
            dim = self.char_lstm_embedding.all_weights[0][idx].size()
            if len(dim) < 2:
                nn.init.constant_(self.char_lstm_embedding.all_weights[0][idx],
                                  1)
            elif len(dim) == 2:
                nn.init.xavier_uniform_(
                    self.char_lstm_embedding.all_weights[0][idx])

    def _init_linear_weights_and_bias(self):
        """Initialize the weight of linear layer using xavier and constant
        with value of one for bias
        """
        # Init linear weights
        nn.init.xavier_uniform_(self.char_linear_embedding.weight)
        # Init bias weights
        nn.init.constant_(self.char_linear_embedding.bias, 1)

    def forward(self, X, X_word):
        word_size = X.size(1)
        char_embeddings = []
        # We have X in dimension of [batch, words, chars]. To use
        # batch calculation we need to loop over all words and
        # calculate the embedding
        for i in range(word_size):
            x = X[:, i, :]
            x = self.char_embedding(x)
            # Need to transpose it to [len, batch, emb] for computational
            # reason
            x = x.transpose(0, 1)
            char_embedding, _ = self.char_lstm_embedding(x)
            # Revert back to [batch, len, emb]
            char_embedding = char_embedding.transpose(0, 1)
            char_embedding = torch.cat(
                [
                    char_embedding[:, 0, :self._char_lstm_hidden_size],
                    char_embedding[:, -1, self._char_lstm_hidden_size:]
                ],
                dim=1)
            char_embedding = self.char_linear_embedding(char_embedding)
            char_embedding = char_embedding.unsqueeze(1)
            char_embeddings.append(char_embedding)

        # Concatenate the whole char embeddings
        final_char_embedding = torch.cat(char_embeddings, dim=1)
        word_embedding = self.word_embedding(X_word)

        # Combine both character and word embeddings
        result = torch.cat([final_char_embedding, word_embedding], 2)
        return result


if __name__ == '__main__':
    x_char = torch.randint(low=0, high=26, size=(5, 3, 16), dtype=torch.long)
    char_len = torch.randint(low=1, high=10, size=(5, 3), dtype=torch.long)
    x_word = torch.randint(low=0, high=100, size=(5, 3), dtype=torch.long)
    lstm_embedding = WordCharLSTMEmbedding()

    lstm_embedding(x_char, char_len, x_word)