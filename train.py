import hydra
import torch

from torch.optim import Adam

import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf
from dataset import get_dataset
from model import get_model_fn

# Set seed for reproducible research
pl.seed_everything(42)


class TaggerModel(pl.LightningModule):
    def __init__(self, hparams, word_vocab, char_vocab, tags_vocab):
        super().__init__()

        self.hparams = hparams

        model_fn = get_model_fn(hparams["embedding"], hparams["tagger"])
        self.model = model_fn(word_vocab, char_vocab, tags_vocab)

    def configure_optimizers(self):
        tagger_params = filter(lambda p: p.requires_grad,
                               self.model.parameters())
        return Adam(lr=self.hparams["training"]["learning_rate"],
                    params=tagger_params)

    def forward(self, char_rep, sentence, sent_len, tags):
        return self.model(char_rep, sentence, sent_len, tags)

    def training_step(self, batch, batch_idx):
        sentence = batch.sentence[0]
        sent_len = batch.sentence[1].numpy()
        char_rep = batch.char_sentence[0]
        tags = batch.tags

        loss = self(char_rep, sentence, sent_len, tags)

        tensorboard_logs = {"train_logs": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            sentence = batch.sentence[0]
            sent_len = batch.sentence[1].numpy()
            char_rep = batch.char_sentence[0]
            tags = batch.tags

            loss = self(char_rep, sentence, sent_len, tags)
            result = torch.tensor(self.model.decode(char_rep, sentence,
                                                    sent_len),
                                  dtype=torch.int32)
            result = result.transpose(1, 0)

            return {
                "loss": loss.item(),
                "predictions": result,
                "targets": tags
            }

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            sentence = batch.sentence[0]
            sent_len = batch.sentence[1].numpy()
            char_rep = batch.char_sentence[0]
            tags = batch.tags

            loss = self(char_rep, sentence, sent_len, tags)
            result = torch.tensor(self.model.decode(char_rep, sentence,
                                                    sent_len),
                                  dtype=torch.int32)
            result = result.transpose(1, 0)

            return {
                "loss": loss.item(),
                "predictions": result,
                "targets": tags
            }


@hydra.main(config_path="config/medium.yaml")
def run(cfg: DictConfig):
    # Dataset
    word_field, char_sentence_field, tags_field, val_iter, train_iter, test_iter = \
        get_dataset(cfg.training.dataset_path, cfg.training.batch_size,
                    pretrained_embedding=cfg.embedding.pretrained)

    # Net initialization
    model = TaggerModel(OmegaConf.to_container(cfg,
                                               resolve=True), word_field.vocab,
                        char_sentence_field.vocab, tags_field.vocab)

    trainer = pl.Trainer()
    trainer.fit(model, train_iter, val_iter)
    trainer.test(model, test_iter)


if __name__ == '__main__':
    run()
