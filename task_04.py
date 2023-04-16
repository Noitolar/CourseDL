import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import utils.nlp as unlp

if __name__ == "__main__":
    config = unlp.config.ConfigObject()

    config.model_class = unlp.nnmodels.TextConvClassifier
    config.model_params = {"num_classes": 2, "dropout_rate": 0.1, "kernel_sizes": [2, 4, 6, 8], "conv_out_channelses": [32, 32, 24, 16],
                           "freeze_embeddings": False, "pretrained_embeddings": unlp.dataset.DatasetSentimentClassifier.get_embeddings_weight()}
    config.device = "cuda:0"
    config.criterion_class = nn.CrossEntropyLoss
    config.criterion_params = {}
    config.log_path = "./logs/movie.conv.log"

    config.dataset = "movie"
    config.seed = 0
    config.sequence_length = 80
    config.batch_size = 32
    config.num_epochs = 4
    config.optimizer_class = optim.AdamW
    config.optimizer_params = {"lr": 0.001, "weight_decay": 1e-4}
    config.scheduler_class = optim.lr_scheduler.StepLR
    config.scheduler_params = {"step_size": 2, "gamma": 0.1}
    config.checkpoint_path = "./checkpoints/movie.conv.pt"

    handler = unlp.handler.ModelHandlerClassifier(config)
    handler.log_config()

    # unlp.dataset.DatasetSentimentClassifier.build_w2v(from_dir="./datasets/movie", to_file="./datasets/movie/vocab.npz", from_pretrained_embeddings_model="./datasets/movie/wiki_word2vec_50.bin")
    trn_set = unlp.dataset.DatasetSentimentClassifier(from_file="./datasets/movie/train.txt", from_vocab="./datasets/movie/vocab.npz", sequence_length=config.sequence_length)
    val_set = unlp.dataset.DatasetSentimentClassifier(from_file="./datasets/movie/validation.txt", from_vocab="./datasets/movie/vocab.npz", sequence_length=config.sequence_length)
    trn_loader = tdata.DataLoader(trn_set, batch_size=config.batch_size, shuffle=True)
    val_loader = tdata.DataLoader(val_set, batch_size=config.batch_size * 8)
    trainer = unlp.trainer.Trainer(handler)

    best_val_accuracy = 0.0
    for epoch in range(config.num_epochs):
        handler.log("    " + "=" * 40)
        trn_report = trainer.train(trn_loader)
        handler.log(f"    [{epoch + 1:03d}] trn-loss: {trn_report['loss']:.4f} --- trn-acc: {trn_report['accuracy']:.2%}")
        val_report = trainer.validate(val_loader)
        handler.log(f"    [{epoch + 1:03d}] val-loss: {val_report['loss']:.4f} --- val-acc: {val_report['accuracy']:.2%}")
        if val_report["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_report["accuracy"]
            if config.checkpoint_path is not None:
                torch.save(handler.model.state_dict, config.checkpoint_path)
    handler.log(f"[=] best-val-acc: {best_val_accuracy:.2%}")
