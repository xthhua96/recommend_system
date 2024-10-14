from tkinter import NO
import pandas as pd
import tensorflow as tf
from keras.layers import Layer
from keras.layers import Dense
from keras.layers import Embedding
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2
from keras import losses
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

"""
    paper: DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
"""


class FMLayer(Layer):
    def __init__(self, k, reg_w, reg_v):
        super().__init__()
        self.k = k
        self.reg_w = reg_w
        self.reg_v = reg_v

    def build(self, input_shape):
        self.w0 = self.add_weight(
            name="w0", shape=(1,), initializer=tf.zeros_initializer, trainable=True
        )
        self.w = self.add_weight(
            name="w",
            shape=(input_shape[-1], 1),
            initializer=tf.random_normal_initializer,
            regularizer=l2(self.reg_w),
            trainable=True,
        )
        self.v = self.add_weight(
            name="v",
            shape=(input_shape[-1], self.k),
            initializer=tf.random_normal_initializer,
            regularizer=l2(self.reg_v),
            trainable=True,
        )

    def call(self, inputs):
        liner_part = tf.matmul(inputs, self.w) + self.w0
        inter_part1 = tf.pow(tf.matmul(inputs, self.v), 2)
        inter_part2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2))
        inter_part = 0.5 * tf.reduce_mean(
            inter_part1 - inter_part2, axis=-1, keepdims=True
        )
        output = 0.5 * (liner_part + inter_part)
        return output


class DenseLayer(Layer):
    def __init__(self, hidden_units, output_dim, activation):
        super().__init__()
        self.hidden_layers = [
            Dense(units=i, activation=activation) for i in hidden_units
        ]
        self.output_layer = Dense(units=output_dim, activation=None)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output


class DeepFM(Model):
    def __init__(
        self, feature_columns, k, reg_w, reg_v, hidden_units, output_dim, activation
    ):
        super().__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            "embed_"
            + str(i): Embedding(
                input_dim=feat["feat_onehot_dim"], output_dim=feat["embed_dim"]
            )
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.FM = FMLayer(k=k, reg_w=reg_w, reg_v=reg_v)
        self.Dense = DenseLayer(
            hidden_units=hidden_units, output_dim=output_dim, activation=activation
        )

    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        # embedding
        sparse_embed = tf.concat(
            [
                self.embed_layers["embed_{}".format(i)](sparse_inputs[:, i])
                for i in range(sparse_inputs.shape[1])
            ],
            axis=1,
        )
        x = tf.concat([dense_inputs, sparse_embed], axis=-1)

        fm_output = self.FM(x)
        dense_output = self.Dense(x)
        output = tf.nn.sigmoid(0.5 * (fm_output + dense_output))
        return output


def create_criteo_dataset(file_path, embed_dim=8, test_size=0.2):
    def sparseFeature(feat, feat_onehot_dim, embed_dim):
        return {
            "feat": feat,
            "feat_onehot_dim": feat_onehot_dim,
            "embed_dim": embed_dim,
        }

    def denseFeature(feat):
        return {"feat": feat}

    data = pd.read_csv(file_path)

    dense_features = ["I" + str(i) for i in range(1, 14)]
    sparse_features = ["C" + str(i) for i in range(1, 27)]

    # 缺失值填充
    data[dense_features] = data[dense_features].fillna(0)
    data[sparse_features] = data[sparse_features].fillna("-1")

    # 归一化
    data[dense_features] = MinMaxScaler().fit_transform(data[dense_features])
    # LabelEncoding编码
    for col in sparse_features:
        data[col] = LabelEncoder().fit_transform(data[col])

    feature_columns = [[denseFeature(feat) for feat in dense_features]] + [
        [
            sparseFeature(feat, data[feat].nunique(), embed_dim)
            for feat in sparse_features
        ]
    ]

    # 数据集划分
    X = data.drop(["label"], axis=1).values
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return feature_columns, (X_train, y_train), (X_test, y_test)


def train(
    data_path: str,
    verbose: bool,
    k: int,
    reg_w: float,
    reg_v: float,
    hidden_units: list,
    output_dim: int,
    activation: str,
    lr: float,
    epochs: int,
    test_size: float,
):
    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(
        file_path=data_path, embed_dim=k, test_size=test_size
    )

    model = DeepFM(
        feature_columns=feature_columns,
        k=k,
        reg_w=reg_w,
        reg_v=reg_v,
        hidden_units=hidden_units,
        output_dim=output_dim,
        activation=activation,
    )

    optimizer = SGD(learning_rate=lr)
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = model(X_train)
            loss = tf.reduce_mean(
                losses.binary_crossentropy(y_true=y_train, y_pred=y_pred)
            )
            if verbose:
                print(f"epoch: {epoch}\tloss: {loss.numpy():.6f}")

        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))

    y_pre = model(X_test)
    y_pre = [1 if x > 0.5 else 0 for x in y_pre]
    acc = accuracy_score(y_true=y_test, y_pred=y_pre)
    print(f"acc: {acc:.6f}")
    return acc
