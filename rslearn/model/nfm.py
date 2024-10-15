import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.layers import Layer
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.models import Model
from keras.optimizers.legacy import SGD
from keras import losses

"""
    Paper: Neural Factorization Machines for Sparse Predictive Analytics
"""


class Dense_layer(Layer):
    def __init__(self, hidden_units, output_dim, activation="relu", dropout=0.0):
        super().__init__()
        self.hidden_layer = [Dense(i, activation=activation) for i in hidden_units]
        self.output_layer = Dense(output_dim, activation=None)
        self.drop_layer = Dropout(rate=dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
        output = self.drop_layer(x)
        return output


class NFM(Model):
    def __init__(
        self, feature_columns, hidden_units, output_dim, activation="relu", dropout=0.0
    ):
        super().__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.dnn_layers = Dense_layer(hidden_units, output_dim, activation, dropout)
        self.emb_layers = {
            "emb_" + str(i): Embedding(feat["feat_onehot_dim"], feat["embed_dim"])
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.bn_layer = BatchNormalization()
        self.output_layer = Dense(1, activation=None)

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        emb = [
            self.emb_layers["emb_" + str(i)](sparse_inputs[:, i])
            for i in range(sparse_inputs.shape[1])
        ]  # list
        emb = tf.convert_to_tensor(emb)  # (26, None, embed_dim)
        emb = tf.transpose(emb, [1, 0, 2])  # (None, 26, embed_dim)

        # Bi-Interaction Layer
        emb = 0.5 * (
            tf.pow(tf.reduce_sum(emb, axis=1), 2)
            - tf.reduce_sum(tf.pow(emb, 2), axis=1)
        )  # (None, embed_dim)
        # Concat
        x = tf.concat([dense_inputs, emb], axis=-1)
        x = self.bn_layer(x)
        x = self.dnn_layers(x)

        outputs = self.output_layer(x)
        return tf.nn.sigmoid(outputs)


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
    hidden_units: list,
    output_dim: int,
    activation: str,
    dropout: float,
    lr: float,
    epochs: int,
    verbose: bool,
):
    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(
        file_path=data_path, embed_dim=embed_dim, test_size=test_size
    )
    model = NFM(
        feature_columns=feature_columns,
        hidden_units=hidden_units,
        output_dim=output_dim,
        activation=activation,
        dropout=dropout,
    )

    optimizer = SGD(learning_rate=lr)
    for i in range(epochs):
        with tf.GradientTape() as tape:
            y_pre = model(X_train)
            loss = tf.reduce_mean(losses.binary_crossentropy(y_train, y_pre))
            grad = tape.gradient(
                loss, model.trainable_variables
            )  # 因为模型涉及bn层，bn层计算的均值和方差不需要计算梯度和更新权重，所以此处需要使用model.trainable_variables
            optimizer.apply_gradients(
                grads_and_vars=zip(grad, model.trainable_variables)
            )  # model.trainable_variables
        if verbose:
            print("epoch: {}, loss: {}".format(i, loss))
    pre = model(X_test)
    pre = [1 if p > 0.5 else 0 for p in pre]
    acc = accuracy_score(y_true=y_test, y_pred=pre)
    print(f"acc: {acc}")
    return acc
