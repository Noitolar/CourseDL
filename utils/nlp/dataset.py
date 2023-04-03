import torch
import torch.utils.data as tdata
import datasets


class DatasetNlpClassifier(tdata.Dataset):
    def __init__(self, tokenizer, max_length, dataset_path, dataset_split, columns_map=None, only_ids=True):
        if columns_map is None:
            columns_map = {"sentences": "sentence", "targets": "label"}

        data = datasets.load_from_disk(dataset_path)[dataset_split]

        # data = data.sort("label")
        # data = data.shuffle(seed=42)
        # xxx = data.select([1, 2, 3, 4, 5, 6, 7, 8])
        # ooo = data.filter(lambda x: x["sentence"].endswith("!"))
        # data_train, data_val = data.train_test_split(test_size = 0.2)
        # data_shard = data.shard(num_shards=10, index=3)
        # data_shard = data_shard.rename_column("sentence", "text")
        # data_shard = data_shard.remove_columns(["label"])
        # pytorch_data = data.set_format(type="torch", columns=["sentence"])

        sentences = data[columns_map["sentences"]]

        tokenizer_params = {
            "text": sentences,
            "max_length": max_length,
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "pt",
            "return_attention_mask": False if only_ids else True,
            "return_token_type_ids": False if only_ids else True,
        }

        self.inputs = tokenizer(**tokenizer_params)
        self.targets = torch.tensor(data[columns_map["targets"]])
        for value in self.inputs.values():
            assert len(value) == len(self.targets)
        print(f"[-] dataset loaded from [{dataset_path}], split [{dataset_split}], [{len(self.targets)}] samples in total.")

    def __getitem__(self, index):
        return {key: self.inputs[key][index] for key in self.inputs.keys()}, self.targets[index]

    def __len__(self):
        return len(self.targets)

    @staticmethod
    def csv_to_arrow(from_csv, to_dir):
        datasets.load_dataset("csv", data_files=from_csv, cache_dir="./tmp").save_to_disk(to_dir)

    @staticmethod
    def arrow_to_csv(from_dir, to_csv):
        datasets.load_from_disk(from_dir).to_csv(to_csv)

    @staticmethod
    def download_from_hub(dataset_name, to_dir, sub_dataset_name=None, cache_dir="./cache"):
        datasets.load_dataset(path=dataset_name, name=sub_dataset_name, cache_dir=cache_dir).save_to_disk(to_dir)


class DatasetNlpGenrator(tdata.Dataset):
    def __init__(self):
        super().__init__()


    def __getitem__(self, index):
        pass
