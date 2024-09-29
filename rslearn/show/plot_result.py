import os

import matplotlib.pyplot as plt
from typing import Optional
from typing import List


def plot_line(
    x_label: str,
    y_label: str,
    save_file_name: str,
    save_dir: str = "./model2code/result",
    x_data: Optional[List] = None,
    y_data: Optional[List] = None,
):
    if x_data is None:
        x_data = []
    if y_data is None:
        y_data = []

    if len(x_data) == 0:
        x_data = range(len(y_data))

    plt.plot(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(os.path.join(save_dir, save_file_name))
    plt.close()
