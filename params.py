import argparse

DATASET_PATH = "data/atis"
CONFIG_BASE_PATH = "config/"
MODEL_PATH = "checkpoints"
SAVE_EVERY_N = 100
SEED = 1234
EARLY_STOPPING_PATIENCE = 10
NUM_RETAINED_MODELS = 5
LEARNING_RATE = 0.001
LOG_INTERVAL = 10
BATCH_SIZE = 16
MAX_EPOCHS = 15

_PARSER = argparse.ArgumentParser(
    description="Google's transformer implementation in PyTorch")
_PARSER.add_argument(
    "--batch_size",
    type=int,
    default=BATCH_SIZE,
    help="Number of batch in single iteration")
_PARSER.add_argument(
    "--dataset_path",
    default=DATASET_PATH,
    help="Path for source training data. Ex: data/train.en")
_PARSER.add_argument(
    "--max_epochs", type=int, default=MAX_EPOCHS, help="Number of epochs")
_PARSER.add_argument(
    "--model_config",
    type=str,
    default=CONFIG_BASE_PATH + "medium.yml",
    help="Location of model config")
_PARSER.add_argument(
    "--learning_rate",
    type=float,
    default=LEARNING_RATE,
    help="Learning rate size")
_PARSER.add_argument(
    "--log_interval",
    type=int,
    default=LOG_INTERVAL,
    help="""Print loss for every N steps""")
_PARSER.add_argument(
    "--save_interval",
    type=int,
    default=SAVE_EVERY_N,
    help="""Save model for every N steps""")
_PARSER.add_argument(
    "--restore_nth_model",
    default="latest",
    help="""Restore the nth model saved on model_path.
    The valid values are string `latest`, and numbers (1, 2, 3, ...)""")
_PARSER.add_argument(
    "--early_stopping_patience",
    default=EARLY_STOPPING_PATIENCE,
    help="""The number of patience required for early stopping""")
_PARSER.add_argument(
    "--num_retained_models",
    type=int,
    default=NUM_RETAINED_MODELS,
    help="""Number of models retained for checkpoint""")
_PARSER.add_argument(
    "--model_path",
    type=str,
    default=MODEL_PATH,
    help="Location to save the model")
ARGS = _PARSER.parse_args()
