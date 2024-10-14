from tabnanny import verbose
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.layers import Layer
from keras.layers import Dense
from keras.layers import Embedding
from keras.models import Model
from keras.optimizers.legacy import SGD
from keras import losses


class Embed_layer(Layer):
    def __init__(self, k, sparse_feature_columns):
        super(Embed_layer, self).__init__()
        self.emb_layers = [
            Embedding(feat["feat_onehot_dim"], k) for feat in sparse_feature_columns
        ]

    def call(self, inputs, **kwargs):
        emb = tf.transpose(
            tf.convert_to_tensor(
                [layer(inputs[:, i]) for i, layer in enumerate(self.emb_layers)]
            ),
            [1, 0, 2],
        )
        emb = tf.reshape(emb, shape=(-1, emb.shape[1] * emb.shape[2]))
        return emb


class Res_layer(Layer):
    def __init__(self, hidden_units):
        super(Res_layer, self).__init__()
        self.dense_layer = [Dense(i, activation="relu") for i in hidden_units]

    def build(self, input_shape):
        self.output_layer = Dense(input_shape[-1], activation=None)

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.dense_layer:
            x = layer(x)
        x = self.output_layer(x)

        output = inputs + x
        return tf.nn.relu(output)


class DeepCrossing(Model):
    def __init__(self, feature_columns, k, hidden_units, res_layer_num):
        super(DeepCrossing, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layer = Embed_layer(k, self.sparse_feature_columns)
        self.res_layer = [Res_layer(hidden_units) for _ in range(res_layer_num)]
        self.output_layer = Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]

        emb = self.embed_layer(sparse_inputs)

        x = tf.concat([dense_inputs, emb], axis=-1)

        for layer in self.res_layer:
            x = layer(x)
        output = self.output_layer(x)
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
    k: int,
    hidden_units: list,
    res_layer_num: int,
    lr: float,
    epochs: int,
    verbose: bool,
):
    feature_columns, (X_train, y_train), (X_test, y_test) = create_criteo_dataset(
        file_path=data_path, embed_dim=embed_dim, test_size=test_size
    )

    model = DeepCrossing(feature_columns, k, hidden_units, res_layer_num)
    optimizer = SGD(learning_rate=lr)

    # train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    #
    # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit(train_dataset, epochs=100)
    # logloss, auc = model.evaluate(X_test, y_test)
    # print('logloss {}\nAUC {}'.format(round(logloss,2), round(auc,2)))

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            pre = model(X_train)
            pre = tf.reshape(pre, shape=(-1, 1))
            loss = tf.reduce_mean(losses.binary_crossentropy(y_train, pre))
        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))
        if verbose:
            print("epoch: {}\tloss: {:.6f}".format(epoch, loss.numpy()))

    pre = model(X_test)
    pre = [1 if p > 0.5 else 0 for p in pre]
    acc = accuracy_score(y_true=y_test, y_pred=pre)
    print(f"acc: {acc}")
    return acc
