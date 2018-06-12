import os
import torch
import glob
import yaml
import six

from ignite.engine import Engine


def load_yaml(config_path):
    if not isinstance(config_path, six.string_types):
        raise ValueError("Got {}, expected string", type(config_path))
    else:
        with open(config_path, "r") as yaml_file:
            config = yaml.load(yaml_file)
            return config


def create_supervised_evaluator(model, inference_fn, metrics={}, cuda=False):
    """
    Factory function for creating an evaluator for supervised models.
    Extended version from ignite's create_supervised_evaluator
    Args:
        model (torch.nn.Module): the model to train
        inference_fn (function): inference function
        metrics (dict of str: Metric): a map of metric names to Metrics
        cuda (bool, optional): whether or not to transfer batch to GPU
            (default: False)
    Returns:
        Engine: an evaluator engine with supervised inference function
    """

    engine = Engine(inference_fn)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_training_function(tagger, opt):
    def training_function(engine, batch):
        tagger.train()
        opt.zero_grad()
        sentence = batch.sentence[0]
        sent_len = batch.sentence[1].numpy()
        char_rep = batch.char_sentence[0]
        tags = batch.tags

        result = tagger(char_rep, sentence, sent_len, tags)
        result.backward()
        opt.step()

        return result.detach()

    return training_function


def create_evaluation_function(tagger):
    def evaluation_function(engine, batch):
        tagger.eval()
        sentence = batch.sentence[0]
        sent_len = batch.sentence[1].numpy()
        char_rep = batch.char_sentence[0]
        tags = batch.tags

        result = torch.tensor(
            tagger.decode(char_rep, sentence, sent_len), dtype=torch.int32)
        result = result.transpose(1, 0)

        return result, tags.detach()

    return evaluation_function


def restore_model(path, restore="latest"):
    """Restore saved model

    Args:
        path (str): The path where the model is saved
        restore (int or str): Amongst the saved model, which last
            saved model would like to be restored. 1, 2, 3, ... or latest

    Returns:
        model: nn.Module
    """

    models = glob.glob(path)
    if len(models) == 0:
        print("No models are found, it's either you put the wrong"
              " path or the model is not even existed yet!")
        return None
    # Sort the models based on the time date
    models.sort(key=os.path.getmtime)
    if restore == "latest":
        restored_model = models[-1]
    else:
        if isinstance(restore, int) and \
                restore < len(models):
            restored_model = models[restore]
        else:
            raise ValueError("Value of restore must be either latest"
                             " or should be an integer with value less than %d"
                             % len(models))

    try:
        model = torch.load(restored_model)
        print("Successfully restored model!")
        return model
    except Exception as e:
        print("Something wrong while restoring the model: %s" % str(e))