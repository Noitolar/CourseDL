import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import transformers as tfm
import random
import numpy as np
import utils.nlp as unlp

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    tfm.set_seed(0)

    model = unlp.nnmodels.Gpt2Generator()
    handler = unlp.handler.ModelHandlerNlpGenerator(model, torch.device("cuda:0"))
    text = "积泥台煤"
    result = handler(text_inputs=text, max_length=80, do_sample=True)
    print(result)
