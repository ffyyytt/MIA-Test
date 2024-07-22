import tensorflow as tf
from classification_models.keras import Classifiers

def getStrategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
    except ValueError:
        tpu = None
        gpus = tf.config.experimental.list_logical_devices("GPU")

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        tf.config.set_soft_device_placement(True)

        print('Running on TPU ', tpu.master())
    elif len(gpus) > 0:
        strategy = tf.distribute.MirroredStrategy(gpus)
        print('Running on ', len(gpus), ' GPU(s) ')
    else:
        strategy = tf.distribute.get_strategy()
        print('Running on CPU')

    print("Number of accelerators: ", strategy.num_replicas_in_sync)

    AUTO = tf.data.experimental.AUTOTUNE
    return strategy, AUTO

class Cosine(tf.keras.layers.Layer):
    def __init__(self, num_classes, scale=32, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.num_classes = num_classes

    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.num_classes, input_shape[-1]), initializer='glorot_uniform', trainable=True)

    def cosine(self, feature):
        x = tf.nn.l2_normalize(feature, axis=1)
        w = tf.nn.l2_normalize(self.W, axis=1)
        cos = tf.matmul(x, tf.transpose(w))
        return cos

    def call(self, inputs):
        feature = inputs
        logits = self.cosine(feature)
        return logits*self.scale

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'scale': self.scale,
            'num_classes': self.num_classes,
        })
        return config

def model_factory(backboneName: str = "resnet18", n_classes: int = 10):
    inputImage = tf.keras.layers.Input(shape = (None, None, 3), dtype=tf.float32, name = f'image')
    backbone, preprocess = Classifiers.get(backboneName)
    feature = tf.keras.layers.GlobalAveragePooling2D(name="feature")(backbone(input_shape = (None, None, 3), weights="imagenet", include_top=False)(inputImage))
    cosine = Cosine(n_classes)(feature)
    output = tf.keras.layers.Softmax(dtype=tf.float32, name = "output")(cosine)

    model = tf.keras.models.Model(inputs = [inputImage], outputs = [output])
    return model, preprocess