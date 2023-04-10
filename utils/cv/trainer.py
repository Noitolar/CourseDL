import torch
import tqdm


class Trainer:
    def __init__(self, handler):
        self.handler = handler
        self.optimizer = handler.config.optimizer_class(handler.model.parameters(), **handler.config.optimizer_params)
        self.scheduler = handler.config.scheduler_class(self.optimizer, **handler.config.scheduler_params) if handler.config.scheduler_class is not None else None

    def train(self, loader, index):
        self.handler.train()
        for inputs, targets in tqdm.tqdm(loader, desc=f"    [{index + 1:03d}] training", delay=0.2, leave=False, ascii="->"):
            preds, loss = self.handler(inputs, targets)
            self.handler.recorder.update(preds, targets, loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        accuracy, loss = self.handler.recorder.accuracy()
        self.handler.log(f"    [{index + 1:03d}] trn-loss: {loss:.4f} --- trn-acc: {accuracy:.2%}")
        self.handler.recorder.clear()
        if self.scheduler is not None:
            self.scheduler.step()
        report = {"index": index, "loss": loss, "accuracy": accuracy}
        return report

    @torch.no_grad()
    def validate(self, loader, index):
        self.handler.eval()
        for inputs, targets in tqdm.tqdm(loader, desc=f"    [{index + 1:03d}] validating", delay=0.2, leave=False, ascii="->"):
            preds, loss = self.handler(inputs, targets)
            self.handler.recorder.update(preds, targets, loss)
        accuracy, loss = self.handler.recorder.accuracy()
        self.handler.log(f"    [{index + 1:03d}] val-loss: {loss:.4f} --- val-acc: {accuracy:.2%}")
        self.handler.recorder.clear()
        report = {"index": index, "loss": loss, "accuracy": accuracy}
        return report

    def train_and_validate(self, trn_loader, val_loader):
        best_val_accuracy = 0.0
        for index in range(self.handler.config.num_epochs):
            self.handler.log("    " + "=" * 40)
            trn_report = self.train(trn_loader, index)
            val_report = self.validate(val_loader, index)
            if val_report["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_report["accuracy"]
                if self.handler.config.checkpoint_path is not None:
                    torch.save(self.handler.model.state_dict, self.handler.config.checkpoint_path)
        self.handler.log(f"[=] best-val-acc: {best_val_accuracy:.2%}")
