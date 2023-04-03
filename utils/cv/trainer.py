import torch
import tqdm


class Trainer:
    def __init__(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, handler, loader, index):
        handler.train()
        for inputs, targets in tqdm.tqdm(loader, desc=f"    [{index + 1:03d}] training", delay=0.2, leave=False, ascii="->"):
            preds, loss = handler(inputs, targets)
            handler.recorder.update(preds, targets, loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        accuracy, loss = handler.recorder.accuracy()
        handler.log(f"    [{index + 1:03d}] trn-loss: {loss:.4f} --- trn-acc: {accuracy:.2%}")
        handler.recorder.clear()
        if self.scheduler is not None:
            self.scheduler.step()
        report = {"index": index, "loss": loss, "accuracy": accuracy}
        return report

    @torch.no_grad()
    def validate(self, handler, loader, index):
        handler.eval()
        for inputs, targets in tqdm.tqdm(loader, desc=f"    [{index + 1:03d}] training", delay=0.2, leave=False, ascii="->"):
            preds, loss = handler(inputs, targets)
            handler.recorder.update(preds, targets, loss)
        accuracy, loss = handler.recorder.accuracy()
        handler.log(f"    [{index + 1:03d}] val-loss: {loss:.4f} --- val-acc: {accuracy:.2%}")
        handler.recorder.clear()
        report = {"index": index, "loss": loss, "accuracy": accuracy}
        return report

    def train_and_validate(self, handler, trn_loader, val_loader, num_epochs, save_to=None):
        best_val_accuracy = 0.0
        for index in range(num_epochs):
            handler.log("    " + "=" * 40)
            trn_report = self.train(handler, trn_loader, index)
            val_report = self.validate(handler, val_loader, index)
            if val_report["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_report["accuracy"]
                if save_to is not None:
                    torch.save(handler.model.state_dict, save_to)
        handler.log(f"[=] best-val-acc: {best_val_accuracy:.2%}")
