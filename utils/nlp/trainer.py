import torch
import tqdm
import utils.nlp as unlp


class Trainer:
    def __init__(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, handler: unlp.handler.ModelHandlerGenerator, loader, index):
        handler.train()
        for inputs, targets in tqdm.tqdm(loader, desc=f"    [{index + 1:03d}] training", delay=0.2, leave=False, ascii="->"):
            preds, loss, hidden = handler(inputs=inputs, targets=targets)
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
    def validate(self, handler: unlp.handler.ModelHandlerGenerator, input_tokens: list, output_length: int):
        handler.eval()
        start_token = handler.encode("<START>")
        end_token = handler.encode("<EOP>")
        if input_tokens[0] != start_token:
            input_tokens.insert(0, start_token)
        output_tokens = input_tokens
        inputs = torch.tensor(input_tokens).unsqueeze(0)
        outputs, _, hiddens = handler(inputs=inputs, hiddens=None)
        for _ in range(output_length - len(input_tokens)):
            preds = outputs[0][-1].argmax(axis=0)
            output_tokens.append(int(preds.item()))
            if preds.item() == end_token:
                break
            else:
                inputs = preds.reshape(1, 1)
                outputs, _, hiddens = handler(inputs=inputs, hiddens=hiddens)
        return "".join([handler.decode(int(x)) for x in output_tokens])
