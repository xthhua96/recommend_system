import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Layer
from keras.layers import Dense
from keras.models import Model
from keras.layers import Embedding
from keras.regularizers import l2
from keras import losses
from keras.optimizers.legacy import SGD
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
    Paper: Wide & Deep Learning for Recommender Systems
"""


class WideLayer(Layer):
    def __init__(self, reg):
        super().__init__()
        self.reg = reg

    def build(self, input_shape):
        self.w0 = self.add_weight(
            name="w0", shape=(1,), initializer=tf.zeros_initializer(), trainable=True
        )
        self.w = self.add_weight(
            name="w",
            shape=(input_shape[-1], 1),
            initializer=tf.random_normal_initializer(),
            trainable=True,
            regularizer=tf.keras.regularizers.l2(self.reg),
        )

    def call(self, inputs, **kwargs):  # 输入为 dense_inputs
        x = tf.matmul(inputs, self.w) + self.w0  # shape: (batchsize, 1)
        return x


class DeepLayer(Layer):
    def __init__(self, hidden_units, output_dim, activation, reg):
        super().__init__()
        self.hidden_layer = [
            Dense(i, activation=activation, kernel_regularizer=l2(reg))
            for i in hidden_units
        ]
        self.output_layer = Dense(output_dim, activation=None)

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
        output = self.output_layer(x)
        return output


class WideDeep(Model):
    def __init__(self, feature_columns, hidden_units, output_dim, activation, reg):
        super().__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embedding_layer = {
            "embed_layer"
            + str(i): Embedding(feat["feat_onehot_dim"], feat["embed_dim"])
            for i, feat in enumerate(self.sparse_feature_columns)
        }

        self.wide = WideLayer(reg=reg)
        self.deep = DeepLayer(hidden_units, output_dim, activation, reg)

    def call(self, inputs):
        # dense_inputs: 数值特征，13维
        # sparse_inputs： 类别特征，26维
        # onehot_inputs：onehot处理的类别特征(wide侧的输入)
        dense_inputs, sparse_inputs, onehot_inputs = (
            inputs[:, :13],
            inputs[:, 13:39],
            inputs[:, 39:],
        )

        # wide部分
        wide_input = tf.concat([dense_inputs, onehot_inputs], axis=-1)
        wide_output = self.wide(wide_input)

        # deep部分
        sparse_embed = tf.concat(
            [
                self.embedding_layer["embed_layer" + str(i)](sparse_inputs[:, i])
                for i in range(sparse_inputs.shape[-1])
            ],
            axis=-1,
        )
        deep_output = self.deep(sparse_embed)

        output = tf.nn.sigmoid(0.5 * (wide_output + deep_output))
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

    y = data["label"]
    X = data.drop(["label"], axis=1)

    # 缺失值填充
    X[dense_features] = X[dense_features].fillna(0)
    X[sparse_features] = X[sparse_features].fillna("-1")

    # 归一化
    X[dense_features] = MinMaxScaler().fit_transform(X[dense_features])

    # Onehot编码(wide侧输入)
    onehot_data = pd.get_dummies(X)

    # LabelEncoding编码(deep侧输入)
    for col in sparse_features:
        X[col] = LabelEncoder().fit_transform(X[col])

    # 拼接到数据集供wide使用
    X = pd.concat([X, onehot_data], axis=1)
    print(X.shape)

    feature_columns = [[denseFeature(feat) for feat in dense_features]] + [
        [sparseFeature(feat, X[feat].nunique(), embed_dim) for feat in sparse_features]
    ]

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=test_size
    )

    return feature_columns, (X_train, y_train), (X_test, y_test)


def train(
    data_path: str,
    lr: float = 1e-2,
    epochs: int = 100,
    test_size: float = 0.2,
    deep_hidden_units=[256, 128, 64],
    output_dim=1,
    activation="relu",
    reg=1e-4,
):

    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(
        data_path, test_size=test_size
    )
    X_train = np.array(X_train).astype("float32")
    X_test = np.array(X_test).astype("float32")

    model = WideDeep(
        feature_columns=feature_columns,
        hidden_units=deep_hidden_units,
        output_dim=output_dim,
        activation=activation,
        reg=reg,
    )

    optimizer = SGD(learning_rate=lr)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_pre = model(X_train)
            loss = tf.reduce_mean(
                losses.binary_crossentropy(y_true=y_train, y_pred=y_pre)
            )
            print(f"Epoch: {epoch}\tLoss: {loss.numpy():.5f}")
        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))
    pre = model(X_test)
    pre = [1 if x > 0.5 else 0 for x in pre]
    acc = accuracy_score(y_true=y_test, y_pred=pre)
    print(f"Accuracy:\t{acc:.2f}")
    return acc
