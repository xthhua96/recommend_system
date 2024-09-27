from tqdm import tqdm

from model2code.model.fm import train
from model2code.show.plot_result import plot_line

if __name__ == "__main__":
    x_data = []
    y_data = []
    for i in tqdm(range(10, 100, 5)):
        accs = [
            train(
                data_path="./model2code/data/train.txt",
                k=i,
                w_reg=1e-5,
                v_reg=1e-5,
                lr=1e-2,
                epochs=100,
            )
            for _ in range(5)
        ]

        x_data.append(i)
        y_data.append(sum(accs) / len(accs))

    plot_line(
        x_label="k",
        y_label="acc",
        save_file_name="fm_k_acc.png",
        x_data=x_data,
        y_data=y_data,
    )
