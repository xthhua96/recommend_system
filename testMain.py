from cgi import test
import time
import functools
from traceback import print_tb

from tqdm import tqdm

from rslearn.model import fm
from rslearn.show.plot_result import plot_line
from rslearn.model import ffm
from rslearn.model import wideanddeep


def metric_helper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()

        # 调用原始函数
        result = func(*args, **kwargs)

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"函数 {func.__name__} 执行时间: {execution_time:.2f} 秒")
        print(f"函数 {func.__name__} 返回结果: {result}")
        return result

    return wrapper


def testFMKAndAcc():
    try:
        x_data = []
        y_data = []
        for i in tqdm(range(10, 100, 5)):
            accs = [
                fm.train(
                    data_path="./rslearn/data/train.csv",
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
    except Exception as e:
        print("error {}".format(e))


@metric_helper
def testFM():
    try:
        acc = fm.train(
            data_path="./rslearn/data/train.csv",
            k=10,
            w_reg=1e-5,
            v_reg=1e-5,
            lr=1e-2,
            epochs=100,
            test_size=0.2,
        )
    except Exception as e:
        print("error {}".format(e))
    return acc


@metric_helper
def testFFM():
    try:
        acc = ffm.train(
            data_path="./rslearn/data/train.csv",
            k=10,
            w_reg=1e-5,
            v_reg=1e-5,
            lr=1e-2,
            epochs=100,
            test_size=0.2,
        )
    except Exception as e:
        print("error {}".format(e))
    return acc


def testWideAndDeep():
    # try:
    acc = wideanddeep.train(
        data_path="./rslearn/data/train.csv",
        lr=1e-2,
        epochs=100,
        test_size=0.2,
        deep_hidden_units=[256, 128, 64],
        output_dim=1,
        activation="relu",
        reg=1e-5,
    )
    # except Exception as e:
    #     print(f"error {e}")
    return acc


if __name__ == "__main__":
    # testFM()
    # testFFM()
    testWideAndDeep()
    pass
