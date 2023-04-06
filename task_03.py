import numpy as np
import utils.nlp as unlp


if __name__ == "__main__":
    dataset = unlp.dataset.DatasetPoemGenerator("poem", 124)
    for x in range(10):
        dataset.show_sentence(x)
