import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.layers import Layer
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers.legacy import SGD
from keras import losses


class FFM_Layer(Layer):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        super(FFM_Layer, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
        self.feature_num = sum(
            [feat["feat_onehot_dim"] for feat in self.sparse_feature_columns]
        ) + len(self.dense_feature_columns)
        self.field_num = len(self.dense_feature_columns) + len(
            self.sparse_feature_columns
        )

    def build(self, input_shape):
        self.w0 = self.add_weight(
            name="w0", shape=(1,), initializer=tf.zeros_initializer(), trainable=True
        )
        self.w = self.add_weight(
            name="w",
            shape=(self.feature_num, 1),
            initializer=tf.random_normal_initializer(),
            regularizer=l2(self.w_reg),
            trainable=True,
        )
        self.v = self.add_weight(
            name="v",
            shape=(self.feature_num, self.field_num, self.k),
            initializer=tf.random_normal_initializer(),
            regularizer=l2(self.v_reg),
            trainable=True,
        )

    def call(self, inputs, **kwargs):
        dense_inputs = inputs[:, :13]
        sparse_inputs = inputs[:, 13:]

        # one-hot encoding
        x = tf.cast(dense_inputs, dtype=tf.float32)
        for i in range(sparse_inputs.shape[1]):
            x = tf.concat(
                [
                    x,
                    tf.one_hot(
                        tf.cast(sparse_inputs[:, i], dtype=tf.int32),
                        depth=self.sparse_feature_columns[i]["feat_onehot_dim"],
                    ),
                ],
                axis=1,
            )

        linear_part = self.w0 + tf.matmul(x, self.w)
        inter_part = 0
        # 每维特征先跟自己的 [field_num, k] 相乘得到Vij*X
        field_f = tf.tensordot(
            x, self.v, axes=1
        )  # [None, 2291] x [2291, 39, 8] = [None, 39, 8]
        # 域之间两两相乘，
        for i in range(self.field_num):
            for j in range(i + 1, self.field_num):
                inter_part += tf.reduce_sum(
                    tf.multiply(field_f[:, i], field_f[:, j]),  # [None, 8]
                    axis=1,
                    keepdims=True,
                )

        return linear_part + inter_part


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


class FFM(Model):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        super(FFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.ffm = FFM_Layer(feature_columns, k, w_reg, v_reg)

    def call(self, inputs, **kwargs):
        output = self.ffm(inputs)
        output = tf.nn.sigmoid(output)
        return output


def train(
    data_path: str,
    k: int,
    w_reg: float = 1e-5,
    v_reg: float = 1e-5,
    lr: float = 1e-2,
    epochs: int = 100,
    test_size: float = 0.2,
):

    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(
        data_path, test_size=test_size
    )

    model = FFM(feature_columns, k=k, w_reg=w_reg, v_reg=v_reg)
    optimizer = SGD(lr)

    for i in range(epochs):
        with tf.GradientTape() as tape:
            y_pre = model(X_train)
            loss = tf.reduce_mean(
                losses.binary_crossentropy(y_true=y_train, y_pred=y_pre)
            )
            # print("loss:\t", loss.numpy())
        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))

    pre = model(X_test)
    pre = [1 if x > 0.5 else 0 for x in pre]
    print("Accuracy: ", accuracy_score(y_test, pre))
    return accuracy_score(y_test, pre)
