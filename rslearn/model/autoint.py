from tabnanny import verbose
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.layers import Layer
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras.optimizers.legacy import SGD
from keras import losses
import tensorflow as tf
import keras.backend as K


class Dense_layer(Layer):
    def __init__(self, hidden_units, activation="relu", dropout=0.0):
        super(Dense_layer, self).__init__()
        self.dense_layer = [Dense(i, activation=activation) for i in hidden_units]
        self.dropout = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.dense_layer:
            x = layer(x)
            x = self.dropout(x)
        return x


class DotProductAttention(Layer):
    def __init__(self, dropout=0.0):
        super(DotProductAttention, self).__init__()
        self._dropout = dropout
        self._masking_num = -(2**32) + 1

    def call(self, inputs):
        # queries: [None, n, k]
        # keys:    [None, n, k]
        # values:  [None, n, k]
        queries, keys, values = inputs
        score = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1]))  # [None, n, n]
        score = score / int(queries.shape[-1]) ** 0.5  # 缩放
        score = K.softmax(score)  # SoftMax
        score = K.dropout(score, self._dropout)  # dropout
        outputs = K.batch_dot(score, values)  # [None, n, k]
        return outputs


class MultiHeadAttention(Layer):
    def __init__(self, n_heads=4, head_dim=64, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._dropout = dropout
        self._att_layer = DotProductAttention(dropout=self._dropout)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self._weights_queries = self.add_weight(
            shape=(input_shape[0][-1], self._n_heads * self._head_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="weights_queries",
        )
        self._weights_keys = self.add_weight(
            shape=(input_shape[1][-1], self._n_heads * self._head_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="weights_keys",
        )
        self._weights_values = self.add_weight(
            shape=(input_shape[2][-1], self._n_heads * self._head_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="weights_values",
        )

    def call(self, inputs):
        # queries: [None, n, k]
        # keys:    [None, n, k]
        # values:  [None, n, k]
        queries, keys, values = inputs
        if self._n_heads * self._head_dim != queries.shape[-1]:
            raise ValueError(
                "n_head * head_dim not equal embedding dim {}".format(queries.shape[-1])
            )

        queries_linear = K.dot(queries, self._weights_queries)  # [None, n, k]
        keys_linear = K.dot(keys, self._weights_keys)  # [None, n, k]
        values_linear = K.dot(values, self._weights_values)  # [None, n, k]

        queries_multi_heads = tf.concat(
            tf.split(queries_linear, self._n_heads, axis=2), axis=0
        )  # [None*n_head, n, k/n_head]
        keys_multi_heads = tf.concat(
            tf.split(keys_linear, self._n_heads, axis=2), axis=0
        )  # [None*n_head, n, k/n_head]
        values_multi_heads = tf.concat(
            tf.split(values_linear, self._n_heads, axis=2), axis=0
        )  # [None*n_head, n, k/n_head]

        att_out = self._att_layer(
            [queries_multi_heads, keys_multi_heads, values_multi_heads]
        )  # [None*n_head, n, k/n_head]
        outputs = tf.concat(
            tf.split(att_out, self._n_heads, axis=0), axis=2
        )  # [None, n, k]
        return outputs


class AutoInt(Model):
    def __init__(
        self,
        feature_columns,
        hidden_units,
        activation="relu",
        dnn_dropout=0.0,
        n_heads=4,
        head_dim=64,
        att_dropout=0.1,
    ):
        super(AutoInt, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.dense_emb_layers = [
            Embedding(feat["feat_onehot_dim"], feat["embed_dim"])
            for feat in self.dense_feature_columns
        ]
        self.sparse_emb_layers = [
            Embedding(feat["feat_onehot_dim"], feat["embed_dim"])
            for feat in self.sparse_feature_columns
        ]
        self.dense_layer = Dense_layer(hidden_units, activation, dnn_dropout)
        self.multi_head_att = MultiHeadAttention(n_heads, head_dim, att_dropout)
        self.out_layer = Dense(1, activation=None)
        k = self.dense_feature_columns[0]["embed_dim"]
        self.W_res = self.add_weight(
            name="W_res",
            shape=(k, k),
            trainable=True,
            initializer=tf.initializers.glorot_normal(),
            regularizer=tf.keras.regularizers.l1_l2(1e-5),
        )

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        # 值为1.0会使embedding报错
        dense_inputs = tf.where(tf.equal(dense_inputs, 1), 0.9999999, dense_inputs)
        dense_emb = [
            layer(dense_inputs[:, i]) for i, layer in enumerate(self.dense_emb_layers)
        ]  # [13, None, k]
        sparse_emb = [
            layer(sparse_inputs[:, i]) for i, layer in enumerate(self.sparse_emb_layers)
        ]  # [26, None, k]
        emb = tf.concat(
            [tf.convert_to_tensor(dense_emb), tf.convert_to_tensor(sparse_emb)], axis=0
        )  # [39, None, k]
        emb = tf.transpose(emb, [1, 0, 2])  # [None, 39, k]

        # DNN
        dnn_input = tf.reshape(
            emb, shape=(-1, emb.shape[1] * emb.shape[2])
        )  # [None, 39*k]
        dnn_out = self.dense_layer(dnn_input)  # [None, out_dim]

        # AutoInt
        att_out = self.multi_head_att([emb, emb, emb])  # [None, 39, k]
        att_out_res = tf.matmul(emb, self.W_res)  # [None, 39, k]
        att_out = att_out + att_out_res
        att_out = tf.reshape(
            att_out, [-1, att_out.shape[1] * att_out.shape[2]]
        )  # [None, 39*k]

        # output
        x = tf.concat([dnn_out, att_out], axis=-1)
        output = self.out_layer(x)
        return tf.nn.sigmoid(output)


def create_criteo_dataset(file_path, embed_dim=64, test_size=0.2):
    def sparseFeature(feat, feat_onehot_dim, embed_dim):
        return {
            "feat": feat,
            "feat_onehot_dim": feat_onehot_dim,
            "embed_dim": embed_dim,
        }

    def denseFeature(feat, embed_dim):
        return {"feat": feat, "feat_onehot_dim": 1, "embed_dim": embed_dim}

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

    feature_columns = [[denseFeature(feat, embed_dim) for feat in dense_features]] + [
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
    activation: str,
    dnn_dropout: float,
    n_heads: int,
    head_dim: int,
    att_dropout: int,
    lr: float,
    batch_size: int,
    epochs: int,
    verbose: bool,
):

    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(
        file_path=data_path, embed_dim=embed_dim, test_size=test_size
    )

    model = AutoInt(
        feature_columns=feature_columns,
        hidden_units=hidden_units,
        activation=activation,
        dnn_dropout=dnn_dropout,
        n_heads=n_heads,
        head_dim=head_dim,
        att_dropout=att_dropout,
    )
    optimizer = SGD(learning_rate=lr)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(batch_size).prefetch(
        tf.data.experimental.AUTOTUNE
    )

    summary_writer = tf.summary.create_file_writer("E:\\PycharmProjects\\tensorboard")
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
            print(f"epoch: {epoch}, loss: {sum(sum_loss)/len(sum_loss):.6f}")

    y_pre = model(X_test)
    y_pre = [1 if x > 0.5 else 0 for x in y_pre]
    acc = accuracy_score(y_true=y_test, y_pred=y_pre)
    print(f"acc: {acc:.6f}")
    return acc
