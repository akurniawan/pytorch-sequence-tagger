import torch
import random

from dataset import get_dataset
from helper import restore_model

BASE_PATH = "data/"
RESTORED_MODEL = "latest"
BATCH_SIZE = 1
SEED = 1234

random.seed(SEED)


def main():
    sentence, char_sentence, tags, _, _, test_iter = \
        get_dataset(BASE_PATH, "atis", BATCH_SIZE, is_inference=True)

    tagger = restore_model(
        "models/ner_cnn-bilstm-crf_*", restore=RESTORED_MODEL)

    final_result = ""
    for it in test_iter:
        words = it.sentence[0]
        sent_len = it.sentence[1]
        char_rep = it.char_sentence[0]

        result = torch.tensor(
            tagger.decode(char_rep, words, sent_len.numpy()),
            dtype=torch.int32)

        sentence_list = words.squeeze(0).numpy().tolist()
        tag_result = result.squeeze(-1).numpy().tolist()

        result_format = "{}  {}\n"
        this_result = ""
        for sent, tag in zip(sentence_list, tag_result):
            this_result += result_format.format(sentence.vocab.itos[sent],
                                                tags.vocab.itos[tag])
        this_result += "\n\n"
        final_result += this_result

    with open(BASE_PATH + "res_atis.txt", "w") as text_file:
        text_file.write(final_result)


if __name__ == '__main__':
    main()