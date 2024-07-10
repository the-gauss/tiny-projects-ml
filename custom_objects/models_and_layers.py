from tensorflow import keras
import tensorflow as tf


class ResidualUnit(keras.layers.Layer):
    def __init__(self, activation, filters, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=strides, padding='same', use_bias=False),
            keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides, padding='same', use_bias=False),
                keras.layers.BatchNormalization()
            ]

    def call(self, inputs, *args, **kwargs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)


class ResNet_34(keras.models.Model):
    def __init__(self, strides=2, activation='relu', input_shape=(224, 224, 3), padding='same',
                 use_bias=False, **kwargs):
        super().__init__()
        prev_filters = 64
        self.residual_layers = []
        for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
            strides = 1 if filters == prev_filters else 2
            self.residual_layers.append(ResidualUnit(filters=filters, activation=activation, strides=strides))
            prev_filters = filters
        self.main_layers = [
            keras.layers.Conv2D(filters=64, kernel_size=7, strides=strides, input_shape=input_shape,
                                padding=padding, use_bias=use_bias),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(activation=activation),
            keras.layers.MaxPool2D(pool_size=3, strides=strides, padding=padding),
            *self.residual_layers,
            keras.layers.GlobalAvgPool2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation='softmax')
        ]

    def call(self, inputs, training=None, mask=None):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        return Z


class LayerNormalization(keras.layers.Layer):
    def __init__(self, units, activation, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.units = int(units)
        self.activation = activation

    def build(self, input_shape):
        self.alpha = self.add_weight(shape=(input_shape[-1]), name='alpha', initializer='ones', trainable=True)
        self.beta = self.add_weight(shape=(input_shape[-1]), name='beta', initializer='zeros', trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, X):
        mean, var = tf.nn.moments(X, axes=[-1], keepdims=True)
        std = tf.sqrt(var + self.epsilon)
        y = self.alpha * ((X - mean) / (std + keras.backend.epsilon)) + self.beta

        if self.activation is not None:
            return self.activation(y)
        return y


# NOT FOR GENERAL APPLICATION, ONLY USE FOR DATASETS IN large_movie_review.ipynb
class TextVectorization(keras.layers.Layer):
    def __init__(self, vocab_size, n_oob=50, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.vocab = None
        self.n_oob = n_oob
        self.lookup_table = None

    def _get_vocab(self, data_sample):
        vocab = set()

        for text, _ in data_sample:
            text = text.numpy()[0].decode('utf-8')
            words = tf.strings.split(text, ' ')

            for word in words:
                vocab.add(word.numpy())
                if len(vocab) >= self.vocab_size:
                    break
        return list(vocab)[:self.vocab_size]

    def adapt(self, data_sample):
        self.vocab = self._get_vocab(data_sample)
        vocab_size = len(self.vocab)
        indices = tf.range(vocab_size, dtype=tf.int64)
        initializer = tf.lookup.KeyValueTensorInitializer(keys=self.vocab, values=indices,
                                                          key_dtype=tf.string, value_dtype=tf.int64)
        self.lookup_table = tf.lookup.StaticVocabularyTable(initializer, num_oov_buckets=self.n_oob)

    def call(self, inputs):
        tokenized_inputs = tf.strings.split(inputs, ' ').to_tensor()
        outputs = self.lookup_table.lookup(tokenized_inputs)
        outputs = tf.one_hot(outputs, depth=self.vocab_size + self.n_oob)
        return outputs
    

# Tokenizer from scratch for plain text, very basic, do not use for production
class TokenizerFromScratch(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def adapt(self, text):
        text = text.lower()
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')

        text = text.split(' ')

        vocab, _ = tf.unique(tf.constant(text))
        indices = tf.range(len(vocab), dtype=tf.int64)

        init_table = tf.lookup.KeyValueTensorInitializer(vocab, indices)
        self.lookup_table = tf.lookup.StaticVocabularyTable(init_table, num_oov_buckets=10)
    
    def call(self, text):
        return self.lookup_table.lookup(text)