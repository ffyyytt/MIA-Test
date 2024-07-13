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

def model_factory(backboneName: str = "resnet18", n_classes: int = 10):
    inputImage = tf.keras.layers.Input(shape = (None, None, 3), dtype=tf.float32, name = f'image')
    backbone, preprocess = Classifiers.get(backboneName)
    feature = tf.keras.layers.GlobalAveragePooling2D()(backbone(input_shape = (None, None, 3), weights=None, include_top=False)(inputImage))
    output = tf.keras.layers.Dense(n_classes, activation='softmax', name='output')(feature)

    model = tf.keras.models.Model(inputs = [inputImage], outputs = [output])
    return model, preprocess