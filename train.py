import torch
import torch.nn as nn
import random

from torch.optim import Adam

from ignite.engine import Events, Engine
from ignite.handlers.checkpoint import ModelCheckpoint
from ignite.handlers.early_stopping import EarlyStopping

from dataset import get_dataset
from model import get_model_fn
from helper import create_supervised_evaluator
from helper import create_training_function
from helper import create_evaluation_function
from helper import restore_model
from helper import load_yaml
from handler import create_log_validation_handler
from handler import create_log_training_loss_handler
from metrics import SequenceTagAccuracy
from params import ARGS, SEED

# Set seed for reproducible research
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def run():
    model_config = load_yaml(ARGS.model_config)
    embedding_config = model_config["embedding"]
    model_config = model_config["tagger"]

    # Dataset
    dataset_config = ARGS.dataset_path.split("/")
    if len(dataset_config) != 2:
        raise ValueError(
            "Dataset path should be in `data_folder/dataset_name` format")
    base_path = dataset_config[0]
    dataset_name = dataset_config[1]

    sentence, char_sentence, tags, val_iter, train_iter, _ = \
        get_dataset(base_path, dataset_name, ARGS.batch_size,
                    pretrained_embedding=embedding_config["pretrained"])

    # Net initialization
    # First, try to load existing model if any
    tagger = restore_model(
        ARGS.model_path + "/ner_cnn-bilstm-crf_*",
        restore=ARGS.restore_nth_model)
    # If none are found, fallback to default initialization
    if not tagger:
        model_fn = get_model_fn(embedding_config, model_config)
        tagger = model_fn(sentence, char_sentence, tags)

    # This will help to automatically registering the model on GPU
    # if one is available
    tagger = nn.DataParallel(tagger)

    tagger_params = filter(lambda p: p.requires_grad, tagger.parameters())
    opt = Adam(lr=ARGS.learning_rate, params=tagger_params)

    # Trainer initialization
    training_fn = create_training_function(tagger, opt)
    evaluation_fn = create_evaluation_function(tagger.module)

    # Create engines for both trainer and evaluator
    trainer = Engine(training_fn)
    evaluator = create_supervised_evaluator(
        model=tagger,
        inference_fn=evaluation_fn,
        metrics={
            "acc": SequenceTagAccuracy(tags.vocab),
        })

    # Handler Initialization
    checkpoint = ModelCheckpoint(
        ARGS.model_path,
        "ner",
        save_interval=ARGS.save_interval,
        n_saved=ARGS.num_retained_models,
        create_dir=True,
        require_empty=False)

    def score_fn(engine):
        print(engine.state.metrics)
        return engine.state.metrics["acc"]

    early_stopper = EarlyStopping(
        trainer=trainer,
        patience=ARGS.early_stopping_patience,
        score_function=score_fn)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, checkpoint,
                              {"cnn-bilstm-crf": tagger.module})
    trainer.add_event_handler(Events.COMPLETED, checkpoint,
                              {"cnn-bilstm-crf": tagger.module})

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED,
        create_log_training_loss_handler(ARGS.log_interval))
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        create_log_validation_handler(evaluator, val_iter))
    # trainer.add_event_handler(Events.EPOCH_COMPLETED, early_stopper)

    trainer.run(train_iter, max_epochs=ARGS.max_epochs)


if __name__ == '__main__':
    run()
