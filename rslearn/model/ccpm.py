import tensorflow as tf
import pandas as pd
from keras.layers import Layer
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Embedding
from keras.optimizers.legacy import SGD
from keras import losses
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class DNN(Layer):
    def __init__(self, hidden_units, out_dim=1, activation="relu", dropout=0.0):
        super(DNN, self).__init__()
        self.dnn_layer = [Dense(i, activation=activation) for i in hidden_units]
        self.out_layer = Dense(out_dim, activation=None)
        self.drop_layer = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.dnn_layer:
            x = layer(x)
            x = self.drop_layer(x)
        output = self.out_layer(x)
        return tf.nn.sigmoid(output)


class KMaxPool(Layer):
    def __init__(self, k):
        super(KMaxPool, self).__init__()
        self.k = k

    def call(self, inputs, **kwargs):
        # inputs: [None, n, k, 1]
        inputs = tf.transpose(inputs, [0, 3, 2, 1])
        k_max = tf.nn.top_k(inputs, k=self.k, sorted=True)[0]
        output = tf.transpose(k_max, [0, 3, 2, 1])
        return output


class CCPM_layer(Layer):
    def __init__(self, filters=[4, 4], kernel_width=[6, 5]):
        super(CCPM_layer, self).__init__()
        self.filters = filters
        self.kernel_width = kernel_width

    def build(self, input_shape):
        n = input_shape[1]
        l = len(self.filters)
        self.conv_layers = []
        self.kmax_layers = []
        for i in range(1, l + 1):
            self.conv_layers.append(
                Conv2D(
                    filters=self.filters[i - 1],
                    kernel_size=(self.kernel_width[i - 1], 1),
                    strides=(1, 1),
                    padding="same",
                    activation="tanh",
                )
            )
            k = (
                max(1, int((1 - pow(i / l, l - i)) * n)) if i < l else 3
            )  # 论文中k随层数衰减
            self.kmax_layers.append(KMaxPool(k=k))
        self.flatten_layer = Flatten()

    def call(self, inputs, **kwargs):
        # inputs: [None, n, k]
        x = tf.expand_dims(inputs, axis=-1)  # [None, n, k, 1]
        for i in range(len(self.filters)):
            x = self.conv_layers[i](x)  # [None, n, k, filters]
            x = self.kmax_layers[i](x)  # [None, n_k, k, filters]
        output = self.flatten_layer(x)  # [None, n_k*k*filters]
        return output


class CCPM(Model):
    def __init__(
        self,
        feature_columns,
        hidden_units,
        out_dim=1,
        activation="relu",
        dropout=0.0,
        filters=[4, 4],
        kernel_width=[6, 5],
    ):
        super(CCPM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.emb_layers = [
            Embedding(feat["feat_onehot_dim"], feat["embed_dim"])
            for i, feat in enumerate(self.sparse_feature_columns)
        ]
        self.dnn_layer = DNN(hidden_units, out_dim, activation, dropout)
        self.ccpm_layer = CCPM_layer(filters, kernel_width)

    def call(self, inputs, training=None, mask=None):
        # dense_inputs:  [None, 13]
        # sparse_inputs: [None, 26]
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        sparse_embed = [
            layer(sparse_inputs[:, i]) for i, layer in enumerate(self.emb_layers)
        ]  # 26 * [None, k]
        sparse_embed = tf.transpose(
            tf.convert_to_tensor(sparse_embed), [1, 0, 2]
        )  # [None, 26, k]

        ccpm_out = self.ccpm_layer(sparse_embed)  # [None, new_field*k]
        x = tf.concat([dense_inputs, ccpm_out], axis=-1)
        output = self.dnn_layer(x)
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
    output_dim: int,
    lr: float,
    activation: str,
    batch_size: int,
    epochs: int,
    test_size: float,
    dropout: float,
    hidden_units: list,
    filters: list,
    kernel_width: list,
    verbose: bool,
):
    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(
        data_path, test_size=test_size
    )

    model = CCPM(
        feature_columns=feature_columns,
        hidden_units=hidden_units,
        out_dim=output_dim,
        activation=activation,
        dropout=dropout,
        filters=filters,
        kernel_width=kernel_width,
    )
    optimizer = SGD(lr)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(batch_size).prefetch(
        tf.data.experimental.AUTOTUNE
    )

    for epoch in range(epochs):
        sum_loss = []
        for batch, data_batch in enumerate(train_dataset):
            X_train, y_train = data_batch[0], data_batch[1]
            with tf.GradientTape() as tape:
                pre = model(X_train)
                loss = tf.reduce_mean(losses.binary_crossentropy(y_train, pre))
                grad = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(zip(grad, model.variables))
            sum_loss.append(loss.numpy())
        if verbose:
            print(
                "epoch: {}\tloss: {:.6f}".format(epoch, sum(sum_loss) / len(sum_loss))
            )

    pre = model(X_test)
    pre = [1 if x > 0.5 else 0 for x in pre]
    acc = accuracy_score(y_true=y_test, y_pred=pre)
    print(f"acc: {acc}")
    return acc
