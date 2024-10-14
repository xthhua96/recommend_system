import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras import losses
from keras.optimizers.legacy import SGD
from sklearn.metrics import accuracy_score


"""
    Paper: Fast Context-aware Recommendationswith Factorization Machines
    关于FM算法原理的推荐讲解：https://www.cnblogs.com/techflow/p/13967844.html
"""


class FM_layer(tf.keras.layers.Layer):
    def __init__(self, k, w_reg, v_reg):
        super(FM_layer, self).__init__()
        self.k = k  # 隐向量vi的维度
        self.w_reg = w_reg  # 权重w的正则项系数
        self.v_reg = v_reg  # 权重v的正则项系数

    def build(self, input_shape):  # 需要根据input来定义shape的变量，可在build里定义)
        self.w0 = self.add_weight(
            name="w0",
            shape=(1,),  # shape:(1,)
            initializer=tf.zeros_initializer(),
            trainable=True,
        )
        self.w = self.add_weight(
            name="w",
            shape=(input_shape[-1], 1),  # shape:(n, 1)
            initializer=tf.random_normal_initializer(),  # 初始化方法
            trainable=True,  # 参数可训练
            regularizer=tf.keras.regularizers.l2(self.w_reg),
        )  # 正则化方法
        self.v = self.add_weight(
            name="v",
            shape=(input_shape[-1], self.k),  # shape:(n, k)
            initializer=tf.random_normal_initializer(),
            trainable=True,
            regularizer=tf.keras.regularizers.l2(self.v_reg),
        )

    def call(self, inputs, **kwargs):
        # inputs维度判断，不符合则抛出异常
        if K.ndim(inputs) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions"
                % (K.ndim(inputs))
            )

        # 线性部分，相当于逻辑回归
        linear_part = tf.matmul(inputs, self.w) + self.w0  # shape:(batchsize, 1)
        # 交叉部分——第一项
        inter_part1 = tf.pow(tf.matmul(inputs, self.v), 2)  # shape:(batchsize, self.k)
        # 交叉部分——第二项
        inter_part2 = tf.matmul(
            tf.pow(inputs, 2), tf.pow(self.v, 2)
        )  # shape:(batchsize, k)
        # 交叉结果
        inter_part = 0.5 * tf.reduce_sum(
            inter_part1 - inter_part2, axis=-1, keepdims=True
        )  # shape:(batchsize, 1)
        # 最终结果
        output = linear_part + inter_part
        return tf.nn.sigmoid(output)  # shape:(batchsize, 1)


class FM(tf.keras.Model):
    def __init__(self, k, w_reg=1e-4, v_reg=1e-4):
        super(FM, self).__init__()
        self.fm = FM_layer(k, w_reg, v_reg)  # 调用写好的FM_layer

    def call(self, inputs, training=None, mask=None):
        output = self.fm(inputs)  # 输入FM_layer得到输出
        return output


def create_criteo_dataset(file_path, test_size=0.3):
    data = pd.read_csv(file_path)
    dense_features = ["I" + str(i) for i in range(1, 14)]  # 数值特征
    sparse_features = ["C" + str(i) for i in range(1, 27)]  # 类别特征

    # 缺失值填充
    data[dense_features] = data[dense_features].fillna(0)
    data[sparse_features] = data[sparse_features].fillna("-1")

    # 归一化（数值特征）
    data[dense_features] = MinMaxScaler().fit_transform(data[dense_features])
    # onehot编码（类别特征）
    data = pd.get_dummies(data)

    # 数据集划分
    X = data.drop(["label"], axis=1).values
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return (X_train, y_train), (X_test, y_test)


def train(
    data_path: str,
    k: int,
    w_reg: float = 1e-5,
    v_reg: float = 1e-5,
    lr: float = 1e-2,
    epochs: int = 100,
    test_size: float = 0.2,
):
    (X_train, y_train), (X_test, y_test) = create_criteo_dataset(
        data_path, test_size=test_size
    )
    # 数据转换为tensor
    X_train = np.where(X_train == True, 1, 0)
    X_test = np.where(X_test == True, 1, 0)

    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    model = FM(k, w_reg, v_reg)
    optimizer = SGD(lr)

    summary_writer = tf.summary.create_file_writer(
        "./model2code/tensorboard/"
    )  # tensorboard可视化文件路径
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_pre = model(X_train)  # 前馈得到预测值
            loss = tf.reduce_mean(
                losses.binary_crossentropy(y_true=y_train, y_pred=y_pre)
            )  # 与真实值计算loss值
            # print("epoch: {} loss: {}".format(epoch, loss.numpy()))
            grad = tape.gradient(loss, model.variables)  # 根据loss计算模型参数的梯度
            optimizer.apply_gradients(
                grads_and_vars=zip(grad, model.variables)
            )  # 将梯度应用到对应参数上进行更新
        # 需要tensorboard记录的变量(不需要可视化可将该模块注释掉)
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=epoch)
    # 评估
    pre = model(X_test)
    pre = [1 if x > 0.5 else 0 for x in pre]
    # print("ACC: ", accuracy_score(y_test, pre))  # ACC: 0.8075
    return accuracy_score(y_test, pre)
