import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import utils.nlp as unlp

if __name__ == "__main__":
    config = unlp.config.ConfigObject()

    config.model_class = unlp.nnmodels.LstmGnerator
    config.model_params = {"vocab_size": 8293, "lstm_input_size": 512, "lstm_output_size": 1024, "num_lstm_layers": 3, "lstm_dropout": 0.5}
    config.device = "cuda:0"
    config.criterion_class = nn.CrossEntropyLoss
    config.criterion_params = {}
    config.dataset_class = unlp.dataset.DatasetPoemGenerator
    config.dataset_params = {"sequence_length": 50, "use_samples": 640}
    config.log_path = f"./logs/poem({config.dataset_params['use_samples']}).lstm.log"
    config.seed = 0
    config.batch_size = 32
    config.num_epochs = 20
    config.optimizer_class = optim.AdamW
    config.optimizer_params = {"lr": 0.002, "weight_decay": 1e-4}
    config.scheduler_class = None
    config.checkpoint_path = f"./checkpoints/poem({config.dataset_params['use_samples']}).lstm.pt"

    handler = unlp.handler.ModelHandlerGenerator(config)
    handler.log_config()

    dataset = config.dataset_class(**config.dataset_params)
    trn_loader = tdata.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    trainer = unlp.trainer.Trainer(handler)

    best_accuracy = 0.0
    best_generation = ""
    for index in range(config.num_epochs):
        handler.log("    " + "=" * 40)
        report = trainer.train(trn_loader, index)
        tokens = trainer.generate(input_tokens=[dataset.encode(x) for x in "风"], output_length=50)
        generation_sample = "".join(dataset.decode(x) for x in tokens)
        handler.log(f"    [{index + 1:03d}] {generation_sample}")
        if report["accuracy"] > best_accuracy:
            best_accuracy = report["accuracy"]
            best_generation = generation_sample
            if config.checkpoint_path is not None:
                torch.save(handler.model.state_dict, config.checkpoint_path)
    handler.log(f"[=] best-acc: {best_accuracy:.2%}")
    handler.log(f"[=] best-generation: {best_generation}")
