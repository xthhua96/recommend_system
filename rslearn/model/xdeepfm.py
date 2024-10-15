import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.layers import Layer
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.models import Model
from keras.optimizers.legacy import SGD
from keras import losses


class Linear(Layer):
    def __init__(self):
        super(Linear, self).__init__()
        self.out_layer = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        output = self.out_layer(inputs)
        return output


class Dense_layer(Layer):
    def __init__(self, hidden_units, out_dim=1, activation="relu", dropout=0.0):
        super(Dense_layer, self).__init__()
        self.hidden_layers = [Dense(i, activation=activation) for i in hidden_units]
        self.out_layer = Dense(out_dim, activation=None)
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        # inputs: [None, n*k]
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.dropout(x)
        output = self.out_layer(x)
        return output


class CIN(Layer):
    def __init__(self, cin_size):
        super(CIN, self).__init__()
        self.cin_size = cin_size  # 每层的矩阵个数

    def build(self, input_shape):
        # input_shape: [None, n, k]
        self.field_num = [input_shape[1]] + self.cin_size  # 每层的矩阵个数(包括第0层)

        self.cin_W = [
            self.add_weight(
                name="w" + str(i),
                shape=(1, self.field_num[0] * self.field_num[i], self.field_num[i + 1]),
                initializer=tf.initializers.glorot_uniform(),
                regularizer=tf.keras.regularizers.l1_l2(1e-5),
                trainable=True,
            )
            for i in range(len(self.field_num) - 1)
        ]

    def call(self, inputs, **kwargs):
        # inputs: [None, n, k]
        k = inputs.shape[-1]
        res_list = [inputs]
        X0 = tf.split(
            inputs, k, axis=-1
        )  # 最后维切成k份，list: k * [None, field_num[0], 1]
        for i, size in enumerate(self.field_num[1:]):
            Xi = tf.split(res_list[-1], k, axis=-1)  # list: k * [None, field_num[i], 1]
            x = tf.matmul(
                X0, Xi, transpose_b=True
            )  # list: k * [None, field_num[0], field_num[i]]
            x = tf.reshape(x, shape=[k, -1, self.field_num[0] * self.field_num[i]])
            # [k, None, field_num[0]*field_num[i]]
            x = tf.transpose(x, [1, 0, 2])  # [None, k, field_num[0]*field_num[i]]
            x = tf.nn.conv1d(input=x, filters=self.cin_W[i], stride=1, padding="VALID")
            # (None, k, field_num[i+1])
            x = tf.transpose(x, [0, 2, 1])  # (None, field_num[i+1], k)
            res_list.append(x)

        res_list = res_list[1:]  # 去掉X0
        res = tf.concat(res_list, axis=1)  # (None, field_num[1]+...+field_num[n], k)
        output = tf.reduce_sum(res, axis=-1)  # (None, field_num[1]+...+field_num[n])
        return output


class xDeepFM(Model):
    def __init__(
        self,
        feature_columns,
        cin_size,
        hidden_units,
        out_dim=1,
        activation="relu",
        dropout=0.0,
    ):
        super(xDeepFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = [
            Embedding(feat["feat_onehot_dim"], feat["embed_dim"])
            for feat in self.sparse_feature_columns
        ]
        self.linear = Linear()
        self.dense_layer = Dense_layer(hidden_units, out_dim, activation, dropout)
        self.cin_layer = CIN(cin_size)
        self.out_layer = Dense(1, activation=None)

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]

        # linear
        linear_out = self.linear(inputs)

        emb = [
            self.embed_layers[i](sparse_inputs[:, i])
            for i in range(sparse_inputs.shape[1])
        ]  # [n, None, k]
        emb = tf.transpose(tf.convert_to_tensor(emb), [1, 0, 2])  # [None, n, k]

        # CIN
        cin_out = self.cin_layer(emb)

        # dense
        emb = tf.reshape(emb, shape=(-1, emb.shape[1] * emb.shape[2]))
        emb = tf.concat([dense_inputs, emb], axis=1)
        dense_out = self.dense_layer(emb)

        output = self.out_layer(linear_out + cin_out + dense_out)
        return tf.nn.sigmoid(output)


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
    cin_size: list,
    hidden_units: list,
    out_dim: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    verbose: bool,
):
    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(
        file_path=data_path, embed_dim=embed_dim, test_size=test_size
    )

    model = xDeepFM(
        feature_columns=feature_columns,
        cin_size=cin_size,
        hidden_units=hidden_units,
        out_dim=out_dim,
        dropout=dropout,
    )
    optimizer = SGD(learning_rate=lr)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(batch_size=batch_size).prefetch(
        tf.data.experimental.AUTOTUNE
    )

    for epoch in range(epochs):
        loss_summary = []
        for batch, data_batch in enumerate(train_dataset):
            X_train, y_train = data_batch[0], data_batch[1]
            with tf.GradientTape() as tape:
                y_pre = model(X_train)
                loss = tf.reduce_mean(
                    losses.binary_crossentropy(y_true=y_train, y_pred=y_pre)
                )
                grad = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))
            loss_summary.append(loss.numpy())
        if verbose:
            print(f"epoch: {epoch}, loss: {sum(loss_summary)/len(loss_summary):.6f}")

    y_pre = model(X_test)
    y_pre = [1 if x > 0.5 else 0 for x in y_pre]
    acc = accuracy_score(y_true=y_test, y_pred=y_pre)
    print(f"acc: {acc:.6f}")
    return acc
