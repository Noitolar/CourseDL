import re
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate


def plot(from_log, num_interpolations=0):
    trn_loss = []
    trn_acc = []
    val_loss = []
    val_acc = []
    with open(from_log, "r", encoding="utf-8") as log:
        pattern = re.compile(r"\d+[.\d]*")
        for line in log.readlines():
            if "---" not in line:
                continue
            datas = list(map(float, re.findall(pattern, line)))
            if "trn" in line:
                trn_loss.append(datas[1])
                trn_acc.append(datas[2])
            elif "val" in line:
                val_loss.append(datas[1])
                val_acc.append(datas[2])

    plt.figure(figsize=(12, 9))
    plt.ylim((0, 1.05 * max(max(trn_loss), max(val_loss))))
    plt.plot(*smooth(np.array(trn_loss), num_interpolations), label="trn-loss", linewidth=3, color="#86cabf")
    plt.plot(*smooth(np.array(val_loss), num_interpolations), label="val-loss", linewidth=3, color="#fa8e7a")
    plt.grid()
    plt.legend()
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 9))
    plt.ylim((0.99 * min(min(trn_acc), min(val_acc))), 1.01 * max(max(trn_acc), max(val_acc)))
    plt.plot(*smooth(np.array(trn_acc), num_interpolations), label="trn-acc", linewidth=3, color="#86cabf")
    plt.plot(*smooth(np.array(val_acc), num_interpolations), label="val-acc", linewidth=3, color="#fa8e7a")
    plt.grid()
    plt.legend()
    plt.show()


def smooth(lst, num_interpolations):
    x = np.linspace(1, len(lst), len(lst))
    y = np.array(lst)
    if num_interpolations <= 0:
        return x, y
    x_smooth = np.linspace(min(x), max(x), num_interpolations)
    smoothing_spline = interpolate.make_smoothing_spline(x, y)
    y_smooth = smoothing_spline(x_smooth)
    return x_smooth, y_smooth


if __name__ == "__main__":
    plot("./logs/dogs_vs_cats.resnet18.log", 200)
