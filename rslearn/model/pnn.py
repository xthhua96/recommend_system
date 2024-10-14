from numpy import broadcast_shapes
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.layers import Layer
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Dropout
import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.models import Model
from keras.optimizers.legacy import SGD
from keras import losses


class DNN_layer(Layer):
    def __init__(self, hidden_units, output_dim, activation="relu", dropout=0.2):
        super().__init__()
        self.hidden_layers = [Dense(i, activation=activation) for i in hidden_units]
        self.output_layer = Dense(output_dim, activation=None)
        self.dropout_layer = Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.dropout_layer(x)
        output = self.output_layer(x)
        return tf.nn.sigmoid(output)


class InnerProductLayer(Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs, **kwargs):  # [None, field, k]
        field_num = inputs.shape[1]

        # for循环计算点乘，复杂度高
        # InnerProduct = []
        # for i in range(field_num - 1):
        #     for j in range(i + 1, field_num):
        #         Inner = inputs[:, i, :] * inputs[:, j, :]  #[None, k]
        #         Inner = tf.reduce_sum(Inner, axis=1, keepdims=True)       #[None, 1]
        #         InnerProduct.append(Inner)
        # InnerProduct = tf.concat(InnerProduct, axis=1)     #[None, field*(field-1)/2]

        # 复杂度更低，先将要相乘的emb找出，存为两个矩阵，然后点乘即可
        row, col = [], []
        for i in range(field_num - 1):
            for j in range(i + 1, field_num):
                row.append(i)
                col.append(j)
        p = tf.transpose(
            tf.gather(tf.transpose(inputs, [1, 0, 2]), row), [1, 0, 2]
        )  # [None, pair_num, k]
        q = tf.transpose(
            tf.gather(tf.transpose(inputs, [1, 0, 2]), col), [1, 0, 2]
        )  # [None, pair_num, k]
        InnerProduct = tf.reduce_sum(p * q, axis=-1)  # [None, pair_num]
        return InnerProduct


class OuterProductLayer(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.field_num = input_shape[1]
        self.k = input_shape[2]
        self.pair_num = self.field_num * (self.field_num - 1) // 2

        # 该形状方便计算,每个外积矩阵对应一个，共pair个w矩阵
        self.w = self.add_weight(
            name="W",
            shape=(self.k, self.pair_num, self.k),
            initializer=tf.random_normal_initializer(),
            regularizer=tf.keras.regularizers.l2(1e-4),
            trainable=True,
        )

    def call(self, inputs, **kwargs):  # [None, field, k]
        row, col = [], []
        for i in range(self.field_num - 1):
            for j in range(i + 1, self.field_num):
                row.append(i)
                col.append(j)
        p = tf.transpose(
            tf.gather(tf.transpose(inputs, [1, 0, 2]), row), [1, 0, 2]
        )  # [None, pair_num, k]
        q = tf.transpose(
            tf.gather(tf.transpose(inputs, [1, 0, 2]), col), [1, 0, 2]
        )  # [None, pair_num, k]
        p = tf.expand_dims(
            p, axis=1
        )  # [None, 1, pair_num, k] 忽略掉第一维，需要两维与w一致才能进行点乘

        tmp = tf.multiply(
            p, self.w
        )  # [None, 1, pair_num, k] * [k, pair_num, k] = [None, k, pair_num, k]
        tmp = tf.reduce_sum(tmp, axis=-1)  # [None, k, pair_num]
        tmp = tf.multiply(tf.transpose(tmp, [0, 2, 1]), q)  # [None, pair_num, k]
        OuterProduct = tf.reduce_sum(tmp, axis=-1)  # [None, pair_num]
        return OuterProduct


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
            self.pool_layers.append(MaxPool2D(pool_size=(self.pooling_width[i], 1)))
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


class PNN(Model):
    def __init__(
        self,
        feature_columns,
        mode,
        hidden_units,
        output_dim,
        activation="relu",
        dropout=0.2,
        use_fgcnn=False,
    ):
        super().__init__()
        self.mode = mode
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.dnn_layer = DNN_layer(hidden_units, output_dim, activation, dropout)
        self.inner_product_layer = InnerProductLayer()
        self.outer_product_layer = OuterProductLayer()
        self.embed_layers = {
            "embed_" + str(i): Embedding(feat["feat_onehot_dim"], feat["embed_dim"])
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.use_fgcnn = use_fgcnn
        if use_fgcnn:
            self.fgcnn_layer = FGCNN_layer()

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]

        # sparse inputs embedding
        embed = [
            self.embed_layers["embed_{}".format(i)](sparse_inputs[:, i])
            for i in range(sparse_inputs.shape[1])
        ]
        embed = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])  # [None, field, k]

        # product之前加入fgcnn层
        if self.use_fgcnn:
            fgcnn_out = self.fgcnn_layer(embed)
            embed = tf.concat([embed, fgcnn_out], axis=1)

        z = embed  # [None, field, k]
        embed = tf.reshape(
            embed, shape=(-1, embed.shape[1] * embed.shape[2])
        )  # [None, field*k]
        # inner product
        if self.mode == "inner":
            inner_product = self.inner_product_layer(z)  # [None, field*(field-1)/2]
            inputs = tf.concat([embed, inner_product], axis=1)
        # outer product
        elif self.mode == "outer":
            outer_product = self.outer_product_layer(z)  # [None, field*(field-1)/2]
            inputs = tf.concat([embed, outer_product], axis=1)
        # inner and outer product
        elif self.mode == "both":
            inner_product = self.inner_product_layer(z)  # [None, field*(field-1)/2]
            outer_product = self.outer_product_layer(z)  # [None, field*(field-1)/2]
            inputs = tf.concat([embed, inner_product, outer_product], axis=1)
        # Wrong Input
        else:
            raise ValueError("Please choice mode's value in 'inner' 'outer' 'both'.")

        output = self.dnn_layer(inputs)
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
    mode: str,  # both, inner, outer
    hidden_units: list,
    output_dim: int,
    activation: str,
    dropout: float,
    use_fgcnn: bool,
    lr: float,
    batch_size: int,
    epochs: int,
    verbose: bool,
):
    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(
        file_path=data_path, embed_dim=embed_dim, test_size=test_size
    )

    model = PNN(
        feature_columns=feature_columns,
        mode=mode,
        hidden_units=hidden_units,
        output_dim=output_dim,
        activation=activation,
        dropout=dropout,
        use_fgcnn=use_fgcnn,
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
            print(
                "epoch: {}\tloss: {:.6f}".format(epoch, sum(sum_loss) / len(sum_loss))
            )

    pre = model(X_test)
    pre = [1 if x > 0.5 else 0 for x in pre]
    acc = accuracy_score(y_true=y_test, y_pred=pre)
    print(f"acc: {acc}")
    return acc
