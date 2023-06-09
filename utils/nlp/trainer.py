import torch
import tqdm
import utils.nlp as unlp


class Trainer:
    def __init__(self, handler: [unlp.handler.ModelHandlerNLP]):
        self.handler = handler
        self.config = handler.config
        self.optimizer = handler.config.optimizer_class(handler.model.parameters(), **handler.config.optimizer_params)
        self.scheduler = handler.config.scheduler_class(self.optimizer, **handler.config.scheduler_params) if handler.config.scheduler_class is not None else None

    def train(self, loader):
        self.handler.train()
        for inputs, targets in tqdm.tqdm(loader, desc=f"    [-] training", delay=0.2, leave=False, ascii="->"):
            preds, loss = self.handler(inputs, targets)
            self.handler.recorder.update(preds, targets, loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        accuracy, loss = self.handler.recorder.accuracy()
        self.handler.recorder.clear()
        if self.scheduler is not None:
            self.scheduler.step()
        report = {"loss": loss, "accuracy": accuracy}
        return report

    @torch.no_grad()
    def validate(self, loader):
        self.handler.eval()
        for inputs, targets in tqdm.tqdm(loader, desc=f"    [-] validating", delay=0.2, leave=False, ascii="->"):
            preds, loss = self.handler(inputs, targets)
            self.handler.recorder.update(preds, targets, loss)
        accuracy, loss = self.handler.recorder.accuracy()
        self.handler.recorder.clear()
        report = {"loss": loss, "accuracy": accuracy}
        return report

    @torch.no_grad()
    def generate(self, input_tokens: list, output_length: int):
        self.handler.eval()
        start_token = 8291
        end_token = 8290
        if input_tokens[0] != start_token:
            input_tokens.insert(0, start_token)
        output_tokens = input_tokens
        inputs = torch.tensor(input_tokens).unsqueeze(0)
        outputs, _, hiddens = self.handler(inputs=inputs, hiddens=None)
        for _ in range(output_length - len(input_tokens)):
            preds = outputs[0][-1].argmax(axis=0)
            output_tokens.append(int(preds.item()))
            if preds.item() == end_token:
                break
            else:
                inputs = preds.reshape(1, 1)
                outputs, _, hiddens = self.handler(inputs=inputs, hiddens=hiddens)
        return output_tokens
