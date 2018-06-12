from torchtext import data


def get_dataset(base_path,
                filename,
                batch_size,
                pretrained_embedding=None,
                is_inference=False):
    sentence = data.Field(lower=False, include_lengths=True, batch_first=True)
    char_nesting = data.Field(lower=False, tokenize=list)
    char_sentence = data.NestedField(char_nesting, include_lengths=True)
    tags = data.Field(batch_first=True)

    train, val, test = data.TabularDataset.splits(
        path=base_path,
        train=filename + ".train.csv",
        validation=filename + ".dev.csv",
        test=filename + ".test.csv",
        format="csv",
        skip_header=True,
        fields=[(("sentence", "char_sentence"), (sentence, char_sentence)),
                ("tags", tags)])
    tags.build_vocab(train.tags)
    if not pretrained_embedding:
        sentence.build_vocab(train.sentence, min_freq=5)
    else:
        sentence.build_vocab(train.sentence, vectors=pretrained_embedding)
    char_sentence.build_vocab(train.char_sentence)

    if is_inference:
        train_iter, val_iter, test_iter = data.Iterator.splits(
            (train, val, test), [batch_size] * 3, repeat=False, sort=False)
    else:
        train_iter, val_iter, test_iter = data.BucketIterator.splits(
            (train, val, test), [batch_size] * 3,
            repeat=False,
            shuffle=True,
            sort_key=lambda x: len(x.sentence),
            sort_within_batch=True)

    return sentence, char_sentence, tags, val_iter, train_iter, test_iter
