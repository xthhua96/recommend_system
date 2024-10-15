import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.layers import Layer
from keras.layers import Dense
from keras.layers import Embedding
from keras.models import Model
from keras.optimizers.legacy import SGD
from keras import losses

import tensorflow as tf


class Interaction_layer(Layer):
    """
    # input shape:  [None, field, k]
    # output shape: [None, field*(field-1)/2, k]
    """

    def __init__(self):
        super().__init__()

    def call(self, inputs, **kwargs):  # [None, field, k]
        element_wise_product_list = []
        for i in range(inputs.shape[1]):
            for j in range(i + 1, inputs.shape[1]):
                element_wise_product_list.append(
                    tf.multiply(inputs[:, i], inputs[:, j])
                )  # [t, None, k]
        element_wise_product = tf.transpose(
            tf.convert_to_tensor(element_wise_product_list), [1, 0, 2]
        )  # [None, t, k]
        return element_wise_product


class Attention_layer(Layer):
    """
    # input shape:  [None, n, k]
    # output shape: [None, k]
    """

    def __init__(self):
        super().__init__()

    def build(self, input_shape):  # [None, field, k]
        self.attention_w = Dense(input_shape[1], activation="relu")
        self.attention_h = Dense(1, activation=None)

    def call(self, inputs, **kwargs):  # [None, field, k]
        x = self.attention_w(inputs)  # [None, field, field]
        x = self.attention_h(x)  # [None, field, 1]
        a_score = tf.nn.softmax(x)
        a_score = tf.transpose(a_score, [0, 2, 1])  # [None, 1, field]
        output = tf.reshape(
            tf.matmul(a_score, inputs), shape=(-1, inputs.shape[2])
        )  # (None, k)
        return output


class AFM_layer(Layer):
    def __init__(self, feature_columns, mode):
        super(AFM_layer, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.mode = mode
        self.embed_layer = {
            "emb_" + str(i): Embedding(feat["feat_onehot_dim"], feat["embed_dim"])
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.interaction_layer = Interaction_layer()
        if self.mode == "att":
            self.attention_layer = Attention_layer()
        self.output_layer = Dense(1)

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        embed = [
            self.embed_layer["emb_" + str(i)](sparse_inputs[:, i])
            for i in range(sparse_inputs.shape[1])
        ]  # list
        embed = tf.convert_to_tensor(embed)
        embed = tf.transpose(embed, [1, 0, 2])  # [None, 26，k]

        # Pair-wise Interaction
        embed = self.interaction_layer(embed)

        if self.mode == "avg":
            x = tf.reduce_mean(embed, axis=1)  # (None, k)
        elif self.mode == "max":
            x = tf.reduce_max(embed, axis=1)  # (None, k)
        else:
            x = self.attention_layer(embed)  # (None, k)

        output = tf.nn.sigmoid(self.output_layer(x))
        return output


class AFM(Model):
    def __init__(self, feature_columns, mode):
        super().__init__()
        self.afm_layer = AFM_layer(feature_columns, mode)

    def call(self, inputs, training=None, mask=None):
        output = self.afm_layer(inputs)
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
        data[col] = LabelEncoder().fit_transform(data[col]).astype(int)

    feature_columns = [[denseFeature(feat) for feat in dense_features]] + [
        [
            sparseFeature(feat, data[feat].nunique(), embed_dim)
            for feat in sparse_features
        ]
    ]

    X = data.drop(["label"], axis=1).values
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return feature_columns, (X_train, y_train), (X_test, y_test)


def train(
    data_path: str,
    embed_dim: int,
    test_size: float,
    mode: str,
    lr: str,
    epochs: int,
    verbose: bool,
):
    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(
        file_path=data_path, embed_dim=embed_dim, test_size=test_size
    )

    model = AFM(feature_columns=feature_columns, mode=mode)
    optimizer = SGD(learning_rate=lr)

    # dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # dataset = dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    #
    # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit(dataset, epochs=100)
    # pre = model.predict(X_test)

    for i in range(epochs):
        with tf.GradientTape() as tape:
            pre = model(X_train)
            loss = tf.reduce_mean(losses.binary_crossentropy(y_train, pre))

        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))
        if verbose:
            print("epoch: {}, loss: {}".format(i, loss))

    pre = model(X_test)
    pre = [1 if p > 0.5 else 0 for p in pre]
    acc = accuracy_score(y_true=y_test, y_pred=pre)
    print(f"acc: {acc}")
    return acc
