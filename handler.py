import numpy as np


def create_log_training_loss_handler(window=10):
    history = []

    def log_training_loss(engine):
        history.append(engine.state.output.numpy())
        if engine.state.iteration % window == 0:
            iterations_per_epoch = len(engine.state.dataloader)
            current_iteration = engine.state.iteration % \
                iterations_per_epoch
            if current_iteration == 0:
                current_iteration = iterations_per_epoch
            avg_loss = np.array(history).mean()
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                  "".format(engine.state.epoch, current_iteration,
                            iterations_per_epoch, avg_loss))
            del history[:]

    return log_training_loss


def create_log_validation_handler(evaluator, val_iter):
    def log_validation_results(engine):
        evaluator.run(val_iter)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["acc"]
        print("=====================================")
        print("Validation Results - Epoch: {}".format(engine.state.epoch))
        print("Avg accuracy: {:.2f}".format(avg_accuracy))
        print("=====================================")

    return log_validation_results
