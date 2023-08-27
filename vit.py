
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from kapre.composed import get_melspectrogram_layer
from tensorflow.keras.layers import TimeDistributed, LayerNormalization
from keras.applications.inception_v3 import InceptionV3

class ClassToken(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value = w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable = True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]

        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls

def mlp(x, cf):
    x = Dense(cf["mlp_dim"], activation="gelu")(x)
    x = Dropout(cf["dropout_rate"])(x)
    x = Dense(cf["hidden_dim"])(x)
    x = Dropout(cf["dropout_rate"])(x)
    return x

def transformer_encoder(x, cf):
    skip_1 = x
    x = LayerNormalization()(x)
    x = MultiHeadAttention(
        num_heads=cf["num_heads"], key_dim=cf["hidden_dim"]
    )(x, x)
    x = Add()([x, skip_1])

    skip_2 = x
    x = LayerNormalization()(x)
    x = mlp(x, cf)
    x = Add()([x, skip_2])

    return x

def AudioViT(cf, params):
    """ Inputs """
    N_CLASSES = params.get('N_CLASSES') or 10
    SR = params.get('SR') or 16000
    DT = params.get('DT') or 1.0

    input_shape = (int(SR*DT), 1)
    i = get_melspectrogram_layer(input_shape=input_shape,
                                 n_mels=128,
                                 pad_end=True,
                                 n_fft=512,
                                 win_length=400,
                                 hop_length=240,
                                 sample_rate=SR,
                                 return_decibel=True,
                                 input_data_format='channels_last',
                                 output_data_format='channels_last')
    x = LayerNormalization(axis=2, name='batch_norm')(i.output)
    x = Conv2D(1, kernel_size=(5,5), strides=(1,1), activation='relu', padding='same')(x)
    print('x.shape', x.shape)

    # input_shape = (cf["num_patches"], cf["patch_size"]*cf["patch_size"]*cf["num_channels"])
    # print(input_shape)
    # inputs = Input(input_shape)     ## (None, 256, 3072)

    """ Patch + Position Embeddings """
    patch_embed = Dense(cf["hidden_dim"])(x)   ## (None, 256, 768)
    patch_embed = Conv2D(cf["hidden_dim"], kernel_size=(5,5), strides=(2,2), activation='relu', padding='same')(patch_embed)
    patch_embed = Conv2D(cf["hidden_dim"], kernel_size=(5,5), strides=(2,2), activation='relu', padding='same')(patch_embed)
    patch_embed = Conv2D(cf["hidden_dim"], kernel_size=(5,5), strides=(2,2), activation='relu', padding='same')(patch_embed)
    patch_embed = Resizing(height = 4, width = 25)(patch_embed)
    print('patch_embed', patch_embed.shape)
    patch_embed = Reshape((cf["num_patches"], cf["hidden_dim"]))(patch_embed)
    print('patch_embed', patch_embed.shape)

    positions = tf.range(start=0, limit=cf["num_patches"], delta=1)
    pos_embed = Embedding(input_dim=cf["num_patches"], output_dim=cf["hidden_dim"])(positions) ## (256, 768)
    embed = patch_embed + pos_embed ## (None, 256, 768)

    """ Adding Class Token """
    token = ClassToken()(embed)
    x = Concatenate(axis=1)([token, embed]) ## (None, 257, 768)

    for _ in range(cf["num_layers"]//2):
        x = transformer_encoder(x, cf)
        # x2 = transformer_encoder(x1, cf)

        # x= tf.expand_dims(x, axis=3)
        # x2= tf.expand_dims(x2, axis=3)

        # x = Concatenate(axis=3)([x, x2])
        # x = Conv2D(1, kernel_size=(5,5), strides=(1,1), activation='relu', padding='same')(x)
        
        # x= tf.squeeze(x, axis=3)
        
    x = tf.expand_dims(x, axis=3)

    x= UpSampling3D(size=(2, 1, 3))(x)
    # x = tf.image.resize(
    #     x,
    #     size,
    #     preserve_aspect_ratio=False,
    #     antialias=False,
    #     name=None
    # )

    x = ResNet50(include_top=False,
                weights="imagenet",
                input_tensor=None,
                input_shape=None,
                pooling='avg',
                classes=None,)(x)

    
    #x = resnet50.ResNet50(x)(x)
    
    #Code Here
    
    # Full Connection
    #print(x.shape)
        

    
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)

    # """ Classification Head """
    # x = LayerNormalization()(x)     ## (None, 257, 768)
    # x = x[:, 0, :]
    # x = Dense(cf["num_classes"], activation="softmax")(x)

    # model = Model(inputs, x)
    o = Dense(N_CLASSES, activation='softmax', name='softmax')(x)
    model = Model(inputs=i.input, outputs=o, name='2d_convolution')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
