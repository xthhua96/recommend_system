import imp
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.layers import Layer
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import Model
from keras import losses
from keras.optimizers.legacy import SGD
import tensorflow as tf


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


class FGCNN_layer(Layer):
    def __init__(
        self,
        filters=[14, 16],
        kernel_width=[7, 7],
        dnn_maps=[3, 3],
        pooling_width=[2, 2],
    ):
        super(FGCNN_layer, self).__init__()
        self.filters = filters
        self.kernel_width = kernel_width
        self.dnn_maps = dnn_maps
        self.pooling_width = pooling_width

    def build(self, input_shape):
        # input_shape: [None, n, k]
        n = input_shape[1]
        k = input_shape[-1]
        self.conv_layers = []
        self.pool_layers = []
        self.dense_layers = []
        for i in range(len(self.filters)):
            self.conv_layers.append(
                Conv2D(
                    filters=self.filters[i],
                    kernel_size=(self.kernel_width[i], 1),
                    strides=(1, 1),
                    padding="same",
                    activation="tanh",
                )
            )
            self.pool_layers.append(MaxPooling2D(pool_size=(self.pooling_width[i], 1)))
        self.flatten_layer = Flatten()

    def call(self, inputs, **kwargs):
        # inputs: [None, n, k]
        k = inputs.shape[-1]
        dnn_output = []
        x = tf.expand_dims(inputs, axis=-1)  # [None, n, k, 1]最后一维为通道
        for i in range(len(self.filters)):
            x = self.conv_layers[i](x)  # [None, n, k, filters[i]]
            x = self.pool_layers[i](x)  # [None, n/poolwidth[i], k, filters[i]]
            out = self.flatten_layer(x)
            out = Dense(self.dnn_maps[i] * x.shape[1] * x.shape[2], activation="relu")(
                out
            )
            out = tf.reshape(out, shape=(-1, out.shape[1] // k, k))
            dnn_output.append(out)
        output = tf.concat(dnn_output, axis=1)  # [None, new_N, k]
        return output


class FGCNN(Model):
    def __init__(
        self,
        feature_columns,
        hidden_units,
        out_dim=1,
        activation="relu",
        dropout=0.0,
        filters=[14, 16],
        kernel_width=[7, 7],
        dnn_maps=[3, 3],
        pooling_width=[2, 2],
    ):
        super(FGCNN, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.emb_layers = [
            Embedding(feat["feat_onehot_dim"], feat["embed_dim"])
            for i, feat in enumerate(self.sparse_feature_columns)
        ]
        self.dnn_layer = DNN(
            hidden_units, out_dim=out_dim, activation=activation, dropout=dropout
        )
        self.fgcnn_layer = FGCNN_layer(
            filters=filters,
            kernel_width=kernel_width,
            dnn_maps=dnn_maps,
            pooling_width=pooling_width,
        )

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

        fgcnn_out = self.fgcnn_layer(sparse_embed)  # [None, new_n, k]
        sparse_embed = tf.concat([sparse_embed, fgcnn_out], axis=1)
        sparse_embed = tf.reshape(
            sparse_embed, shape=[-1, sparse_embed.shape[1] * sparse_embed.shape[2]]
        )

        input = tf.concat([dense_inputs, sparse_embed], axis=-1)
        output = self.dnn_layer(input)
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
    embed_dim: int,
    test_size: float,
    hidden_units: list,
    output_dim: int,
    activation: str,
    dropout: float,
    filters: list,
    kernel_width: list,
    dnn_maps: list,
    pooling_width: list,
    lr: float,
    batch_size: int,
    epochs: int,
    verbose: bool,
):
    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(
        file_path=data_path, embed_dim=embed_dim, test_size=test_size
    )

    model = FGCNN(
        feature_columns,
        hidden_units=hidden_units,
        out_dim=output_dim,
        activation=activation,
        dropout=dropout,
        filters=filters,
        kernel_width=kernel_width,
        dnn_maps=dnn_maps,
        pooling_width=pooling_width,
    )
    optimizer = SGD(learning_rate=lr)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(batch_size=batch_size).prefetch(
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
            print("epoch:{}, loss:{}".format(epoch, sum(sum_loss) / len(sum_loss)))

    pre = model(X_test)
    pre = [1 if x > 0.5 else 0 for x in pre]
    print("Accuracy: ", accuracy_score(y_test, pre))
    return accuracy_score(y_test, pre)
