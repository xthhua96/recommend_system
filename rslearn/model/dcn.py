import pandas as pd
import tensorflow as tf
from keras.layers import Layer
from keras.layers import Dense
from keras.models import Model
from keras.layers import Embedding
from keras.regularizers import l2
from keras.optimizers import SGD
from keras import losses
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


"""
    Paper: Deep_Crossing_Web_Scale_Modeling_Without_Manually_Crafted_Combinatorial_Features
"""


class DenseLayer(Layer):
    def __init__(self, hidden_units, output_dim, activation):
        super().__init__()
        self.hidden_layer = [
            Dense(hidden_dim, activation=activation) for hidden_dim in hidden_units
        ]
        self.output_layer = Dense(output_dim, activation=None)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
        output = self.output_layer(x)
        return output


class CrossLayer(Layer):
    def __init__(self, layer_num, reg_w, reg_b):
        super().__init__()
        self.layer_num = layer_num
        self.reg_w = reg_w
        self.reg_b = reg_b

    def build(self, input_shape):
        self.cross_weight = [
            self.add_weight(
                name="cross_weight_" + str(i),
                shape=(input_shape[-1], 1),
                initializer=tf.random_normal_initializer(),
                regularizer=l2(self.reg_w),
                trainable=True,
            )
            for i in range(self.layer_num)
        ]
        self.cross_bias = [
            self.add_weight(
                name="cross_bias_" + str(i),
                shape=(input_shape[-1], 1),
                initializer=tf.zeros_initializer(),
                regularizer=l2(self.reg_b),
                trainable=True,
            )
            for i in range(self.layer_num)
        ]

    def call(self, inputs, *args, **kwargs):
        x0 = tf.expand_dims(inputs, axis=2)
        xl = x0
        for i in range(self.layer_num):
            xl_w = tf.matmul(tf.transpose(xl, [0, 2, 1]), self.cross_weight[i])
            xl = tf.matmul(x0, xl_w) + self.cross_bias[i] + xl

        output = tf.squeeze(xl, axis=2)
        return output


class DCN(Model):
    def __init__(
        self,
        feature_columns,
        hidden_units,
        output_dim,
        activation,
        layer_num,
        reg_w,
        reg_b,
    ):
        super().__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            "embed_" + str(i): Embedding(feat["feat_onehot_dim"], feat["embed_dim"])
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.dense_layer = DenseLayer(
            hidden_units=hidden_units, output_dim=output_dim, activation=activation
        )
        self.cross_layer = CrossLayer(layer_num=layer_num, reg_w=reg_w, reg_b=reg_b)
        self.output_layer = Dense(units=1, activation=None)

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
        x = tf.concat([dense_inputs, sparse_embed], axis=1)
        # Crossing layer
        cross_output = self.cross_layer(x)
        # Dense layer
        dnn_output = self.dense_layer(x)
        x = tf.concat([cross_output, dnn_output], axis=1)
        output = tf.nn.sigmoid(self.output_layer(x))
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
    verbose: bool,
    reg_w: float = 1e-5,
    reg_b: float = 1e-5,
    layer_num: int = 6,
    hidden_units=[256, 128, 64],
    output_dim=1,
    lr: float = 1e-2,
    activation="relu",
    epochs: int = 100,
    test_size: float = 0.2,
):
    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(
        data_path, test_size=test_size
    )
    model = DCN(
        feature_columns=feature_columns,
        hidden_units=hidden_units,
        output_dim=output_dim,
        activation=activation,
        layer_num=layer_num,
        reg_w=reg_w,
        reg_b=reg_b,
    )
    optimizer = SGD(learning_rate=lr)
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_pre = model(X_train)
            loss = tf.reduce_mean(
                losses.binary_crossentropy(y_true=y_train, y_pred=y_pre)
            )
        if verbose:
            print("epoch: {}\tloss: {:.6f}".format(epoch, loss.numpy()))

        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
    pre = model(X_test)
    pre = [1 if p > 0.5 else 0 for p in pre]
    acc = accuracy_score(y_true=y_test, y_pred=pre)
    print(f"acc: {acc}")
    return acc
