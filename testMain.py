from cgi import test
import time
import functools

from tqdm import tqdm

from rslearn.model import autoint, fm
from rslearn.show.plot_result import plot_line
from rslearn.model import ffm
from rslearn.model import wideanddeep
from rslearn.model import dcn
from rslearn.model import deepfm
from rslearn.model import ccpm
from rslearn.model import pnn
from rslearn.model import dc
from rslearn.model import afm
from rslearn.model import nfm
from rslearn.model import xdeepfm
from rslearn.model import autoint


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


@metric_helper
def testWideAndDeep():
    try:
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
    except Exception as e:
        print(f"error {e}")
    return acc


@metric_helper
def testDCN():
    # try:
    acc = dcn.train(
        data_path="./rslearn/data/train.csv",
        verbose=True,
        reg_w=1e-5,
        reg_b=1e-5,
        layer_num=6,
        hidden_units=[256, 128, 64],
        output_dim=1,
        lr=0.01,
        activation="relu",
        epochs=100,
        test_size=0.2,
    )
    # except Exception as e:
    #     print(f"error {e}")
    return acc


@metric_helper
def testDeepFM():
    # try:
    acc = deepfm.train(
        data_path="./rslearn/data/train.csv",
        verbose=True,
        k=8,
        reg_w=1e-5,
        reg_v=1e-5,
        hidden_units=[256, 128, 64],
        output_dim=1,
        activation="relu",
        lr=0.01,
        epochs=100,
        test_size=0.2,
    )
    # except Exception as e:
    #     print(f"error {e}")
    return acc


@metric_helper
def testCCPM():
    acc = ccpm.train(
        data_path="./rslearn/data/train.csv",
        output_dim=1,
        lr=1e-2,
        activation="relu",
        batch_size=32,
        epochs=100,
        test_size=0.2,
        dropout=0.2,
        hidden_units=[128],
        filters=[4, 4],
        kernel_width=[6, 5],
        verbose=True,
    )
    return acc


@metric_helper
def testPNN():
    acc = pnn.train(
        data_path="./rslearn/data/train.csv",
        embed_dim=8,
        test_size=0.2,
        mode="inner",
        hidden_units=[256, 128, 64],
        output_dim=1,
        activation="relu",
        dropout=0.2,
        use_fgcnn=True,
        lr=1e-2,
        batch_size=32,
        epochs=100,
        verbose=True,
    )
    return acc


@metric_helper
def testDC():
    acc = dc.train(
        data_path="./rslearn/data/train.csv",
        embed_dim=8,
        test_size=0.2,
        k=32,
        hidden_units=[256, 256],
        res_layer_num=4,
        lr=1e-2,
        epochs=100,
        verbose=True,
    )
    return acc


@metric_helper
def testAFM():
    acc = afm.train(
        data_path="./rslearn/data/train.csv",
        embed_dim=8,
        test_size=0.2,
        mode="att",  # avg, max
        lr=1e-2,
        epochs=100,
        verbose=False,
    )
    return acc


@metric_helper
def testNFM():
    acc = nfm.train(
        data_path="./rslearn/data/train.csv",
        embed_dim=8,
        test_size=0.2,
        hidden_units=[256, 128, 64],
        output_dim=1,
        activation="relu",
        dropout=0.3,
        lr=1e-2,
        epochs=100,
        verbose=True,
    )
    return acc


@metric_helper
def testxDeepFM():
    acc = xdeepfm.train(
        data_path="./rslearn/data/train.csv",
        embed_dim=8,
        test_size=0.2,
        cin_size=[128, 128],
        hidden_units=[256, 128, 64],
        out_dim=1,
        dropout=0.5,
        lr=1e-3,
        epochs=100,
        batch_size=64,
        verbose=True,
    )
    return acc


@metric_helper
def testAutoInt():
    acc = autoint.train(
        data_path="./rslearn/data/train.csv",
        embed_dim=64,
        test_size=0.2,
        hidden_units=[256, 128, 64],
        activation="relu",
        dnn_dropout=0.2,
        n_heads=4,
        head_dim=16,
        att_dropout=0.2,
        lr=1e-2,
        batch_size=32,
        epochs=100,
        verbose=True,
    )
    return acc


if __name__ == "__main__":
    # testFM()
    # testFFM()
    # testWideAndDeep()
    # testDCN()
    # testDeepFM()
    # testCCPM()
    # testPNN()
    # testDC()
    # testAFM()
    # testNFM()
    # testxDeepFM()
    testAutoInt()
    pass
