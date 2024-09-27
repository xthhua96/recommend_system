import time
import functools

from tqdm import tqdm

from model2code.model import fm
from model2code.show.plot_result import plot_line
from model2code.model import ffm


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
    except Exception as e:
        print("error {}".format(e))


@metric_helper
def testFM():
    try:
        acc = fm.train(
            data_path="./model2code/data/train.txt",
            k=10,
            w_reg=1e-5,
            v_reg=1e-5,
            lr=1e-2,
            epochs=100,
        )
    except Exception as e:
        print("error {}".format(e))
    return acc


@metric_helper
def testFFM():
    try:
        acc = ffm.train(
            data_path="./model2code/data/train.txt",
            k=10,
            w_reg=1e-5,
            v_reg=1e-5,
            lr=1e-2,
            epochs=100,
        )
    except Exception as e:
        print("error {}".format(e))
    return acc


if __name__ == "__main__":
    # testFM()
    testFFM()
    pass
