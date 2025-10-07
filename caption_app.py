import tensorflow as tf
import tensorflow.keras as keras
import os
import json
import re
# import cv2
from abc import abstractmethod, ABC
import time
import numpy as np
import random
import itertools
import math
from itertools import accumulate
from functools import reduce
import pandas as pd

IDEAL_SHAPE = (256, 256, 3)
BATCH_SIZE = 25 if tf.config.list_physical_devices('GPU') else 2
MAXIMUM_LENGTH = 70
EXPANSION_LENGTH = 300
NUM_CAPTIONS = 5

strategy = tf.distribute.get_strategy()

def load_tokenizer():
    with open('tokenizer.json') as f:
        return json.load(f)


tokenizer = tokenize_dict = load_tokenizer()

INPUT_SEQ_LENGTH = MAXIMUM_LENGTH
TEXT_EMBEDDING_DIM = 768
VOCAB_SIZE = 4000
MESHED_BUFFER_SIZE = 100
MESHED_DEPTH = 4
ATTENTION_HEAD = 16
ATTENTION_CHOICE = 3
DECODER_ATTENTION_CHOICE = 2
BACKBONE_CHOICE = 3

model_name = {
    0: 'RVSA',
    1: 'Meshed_memory',
    2: 'Zero_mesh',
    3: 'Static_attention',
    4: 'Captured_static_attention',
}[ATTENTION_CHOICE]

decoder_model = {
    0: 'Mesh_decoder',
    1: 'T5',
    2: 'No_mesh',
    3: 'Dynamic_attention',
}[DECODER_ATTENTION_CHOICE]

vision_name = {
    0: 'Resnet152',
    1: 'Resnet50',
    2: 'VGG16',
    3: 'EfficientNetB2',
    4: 'Short',
    5: 'Swin',
    6: 'Inception',
    7: 'MobileNet'
}[BACKBONE_CHOICE]

class Mask(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs, **kwargs):
        length = tf.shape(inputs)[-2]
        mask = tf.sequence_mask(tf.range(length + 1)[1:], length)
        return tf.where(mask, 0., -1e9)


class SeqEmbedding(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.token_emb = keras.layers.Embedding(
            VOCAB_SIZE,
            TEXT_EMBEDDING_DIM)
        self.position = keras.layers.Embedding(
            INPUT_SEQ_LENGTH,
            TEXT_EMBEDDING_DIM
        )

    def call(self, txt):
        return self.token_emb(txt) + self.position(tf.range(tf.shape(txt)[-1]))[tf.newaxis, ...]


class Attention(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.Q = keras.layers.Dense(TEXT_EMBEDDING_DIM)
        self.K = keras.layers.Dense(TEXT_EMBEDDING_DIM)
        self.V = keras.layers.Dense(TEXT_EMBEDDING_DIM)

    def call(self, q, k, v, mask=None, **kwargs):
        Q = self.Q(q)
        K = self.K(k)
        V = self.V(v)
        attention = tf.matmul(Q, K, transpose_b=True) / (TEXT_EMBEDDING_DIM ** 0.5)
        if mask is not None:
            attention += mask
        attention = tf.nn.softmax(attention, axis=-1)
        return tf.matmul(attention, V)


class SelfAttention(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.sa = Attention()
        self.norm = keras.layers.LayerNormalization()
        self.mask = Mask()

    def call(self, inputs, **kwargs):
        mask = self.mask(inputs)
        return self.norm(self.sa(inputs, inputs, inputs, mask) + inputs)


class CrossAttention(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.ca = Attention()
        self.norm = keras.layers.LayerNormalization()

    def call(self, src, mem, **kwargs):
        return self.norm(self.ca(src, mem, mem) + src)


class DecoderLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.sa = SelfAttention()
        self.ca = CrossAttention()
        self.ff = FeedForward()
        self.norm = keras.layers.LayerNormalization()
        self.mask = Mask()

    def call(self, inp):
        seq, enc = inp
        sa = self.sa(seq)
        ca = self.ca(sa, enc)
        return self.ff(ca)


class EncoderLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.attn = Attention()
        self.norm = keras.layers.LayerNormalization()

    def call(self, inputs):
        out = self.attn(inputs, inputs, inputs) + inputs
        return self.norm(out)


class FeedForward(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.seq = keras.Sequential([
            keras.layers.Dense(TEXT_EMBEDDING_DIM * 2, activation='relu'),
            keras.layers.Dense(TEXT_EMBEDDING_DIM),
            keras.layers.Dropout(0.3),
        ])
        self.norm = keras.layers.LayerNormalization()

    def call(self, inputs, **kwargs):
        return self.norm(self.seq(inputs) + inputs)


class FastCaption(keras.Model):
    def __init__(self):
        super().__init__()
        self.backbone = keras.applications.VGG16(include_top=False)
        self.backbone.trainable = False
        self.decoder = DecoderLayer()
        self.encoder = EncoderLayer()
        self.adapt = keras.layers.Dense(TEXT_EMBEDDING_DIM)
        self.dense = keras.layers.Dense(VOCAB_SIZE, activation='softmax')
        self.embedding = SeqEmbedding()

    def call(self, inputs):
        img, txt = inputs
        img = keras.applications.vgg16.preprocess_input(img)
        img = self.backbone(img) # (batch, 8, 8, 2048)
        img = self.adapt(img) # (batch, 8, 8, 768)
        img = tf.reshape(img, [tf.shape(img)[0], tf.shape(img)[1] * tf.shape(img)[2], tf.shape(img)[3]])
        img = self.encoder(img)
        seq = self.embedding(txt) # (b, p, length, dim)
        out = self.decoder((seq, img[:, tf.newaxis]))
        return self.dense(out)

    @staticmethod
    @tf.function
    def generate_caption(model, img):
        img = keras.applications.vgg16.preprocess_input(img)
        img = model.backbone(img)
        img = model.adapt(img)
        img = tf.reshape(img, [tf.shape(img)[0], tf.shape(img)[1] * tf.shape(img)[2], tf.shape(img)[3]])
        img = model.encoder(img)
        txt = tf.zeros((1, 1, INPUT_SEQ_LENGTH), dtype=tf.int32)
        return FastCaption._generate_from_seed(model, img, txt, tf.constant(0, dtype=tf.int32))

    @staticmethod
    @tf.function
    def _generate_from_seed(model, img, txt, index):
        while tf.math.logical_not(tf.math.logical_or(tf.math.reduce_any(tf.equal(txt, 2)), tf.math.reduce_all(tf.not_equal(txt, 0)))):
            seq = model.embedding(txt)
            out = model.decoder((seq, img))
            prob = model.dense(out)
            new_text = tf.argmax(prob, axis=-1, output_type=tf.int32)
            valid = tf.cast(tf.range(tf.shape(txt)[2]) <= index, dtype=tf.int32)
            new_text = new_text * valid
            txt = tf.concat([tf.ones((1, 1, 1), dtype=tf.int32), new_text[:, :, :-1]], axis=2)
            index = index + 1
        return tf.concat([txt[:, :, 1:], tf.zeros((1, 1, 1), dtype=tf.int32)], axis=2)


class MemorizedAttention(keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.Q = keras.layers.Dense(dim)
        self.K = keras.layers.Dense(dim)
        self.V = keras.layers.Dense(dim)
        self.dim = dim
        self.mask = Mask()


    def build(self, input_shape):
        self.memory_k = self.add_weight(
            shape=(1, 1, EXPANSION_LENGTH, self.dim),
            initializer="glorot_uniform",
            trainable=True,
            name="memory_k"
        )
        self.memory_v = self.add_weight(
            shape=(1, 1, EXPANSION_LENGTH, self.dim),
            initializer="glorot_uniform",
            trainable=True,
            name="memory_v"
        )

    def call(self, q, k, v, use_causal=False):
        repeater = tf.concat([tf.shape(q)[:-2], tf.constant([1, 1], dtype=tf.int32)], axis=-1)
        memory_k = tf.tile(self.memory_k, repeater)
        memory_v = tf.tile(self.memory_v, repeater)
        Q = self.Q(q)
        K = tf.concat([self.K(k), memory_k], axis=-2)
        V = tf.concat([self.V(v), memory_v], axis=-2)
        attn = tf.matmul(Q, K, transpose_b=True) / math.sqrt(TEXT_EMBEDDING_DIM)
        if use_causal:
            mask = self.mask(attn)
            attn += mask
        attn = tf.nn.softmax(attn, axis=-1)
        return tf.matmul(attn, V)


class MultiheadMemorizedAttention(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.heads = [MemorizedAttention(TEXT_EMBEDDING_DIM // ATTENTION_HEAD) for _ in range(ATTENTION_HEAD)]
        self.norm = keras.layers.LayerNormalization()
        self.dense = keras.layers.Dense(TEXT_EMBEDDING_DIM)

    def call(self, q, k, v, use_causal=False):
        tensors = [head(q, k, v, use_causal=use_causal) for head in self.heads]
        return self.norm(self.dense(tf.concat(tensors, axis=-1)) + q)


class RelativePositionalSelfAttention(keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.seq_length = 81
        self.positional_k = keras.layers.Embedding(
            self.seq_length * 2 + 1,
            self.dim
        )

    def embedding_net(self, Q):
        seq_length = tf.shape(Q)[-2]
        return tf.range(seq_length - 1, seq_length * 2 - 1)[..., tf.newaxis] - tf.range(seq_length)

    def mask(self, attn):
        mask = tf.range(tf.shape(attn)[-2])[..., tf.newaxis] >= tf.range(tf.shape(attn)[-2])
        return tf.where(mask, 0., -1e9)

    def call(self, Q, K, V, use_causal=False):
        emb = self.positional_k(self.embedding_net(Q))
        eij = tf.matmul(Q, K, transpose_b=True) + tf.einsum('...id, ijd -> ...ij', Q, emb)
        attn = eij / math.sqrt(TEXT_EMBEDDING_DIM)
        if use_causal:
            mask = self.mask(attn)
            attn += mask
        attn = tf.nn.softmax(attn, axis=-1)
        return tf.matmul(attn, V)


class MultiheadRelativePositionalSelfAttention(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.heads = [RelativePositionalSelfAttention(TEXT_EMBEDDING_DIM // ATTENTION_HEAD) for _ in range(ATTENTION_HEAD)]
        self.Q = keras.layers.Dense(TEXT_EMBEDDING_DIM)
        self.K = keras.layers.Dense(TEXT_EMBEDDING_DIM)
        self.V = keras.layers.Dense(TEXT_EMBEDDING_DIM)

    def call(self, q, k, v, use_causal=False):
        Q = self.Q(q)
        K = self.K(k)
        V = self.V(v)
        splitted_Q = tf.split(Q, num_or_size_splits=ATTENTION_HEAD, axis=-1)
        splitted_K = tf.split(K, num_or_size_splits=ATTENTION_HEAD, axis=-1)
        splitted_V = tf.split(V, num_or_size_splits=ATTENTION_HEAD, axis=-1)
        tensors = [head(q, k, v, use_causal=use_causal) for head, q, k, v in zip(self.heads, splitted_Q, splitted_K, splitted_V)]
        return tf.concat(tensors, axis=-1)


class NormalAttention(keras.layers.Layer):
    def __init__(self, dim):
        self.dim = dim
        super().__init__()

    def mask(self, attn):
        mask = tf.range(tf.shape(attn)[-2])[..., tf.newaxis] >= tf.range(tf.shape(attn)[-2])
        return tf.where(mask, 0., -1e9)

    def call(self, Q, K, V, use_causal=False):
        eij = tf.matmul(Q, K, transpose_b=True)
        attn = eij / math.sqrt(self.dim)
        if use_causal:
            mask = self.mask(attn)
            attn += mask
        attn = tf.nn.softmax(attn, axis=-1)
        return tf.matmul(attn, V)


class MultiheadAttention(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.heads = [NormalAttention(TEXT_EMBEDDING_DIM // ATTENTION_HEAD) for _ in range(ATTENTION_HEAD)]
        self.Q = keras.layers.Dense(TEXT_EMBEDDING_DIM)
        self.K = keras.layers.Dense(TEXT_EMBEDDING_DIM)
        self.V = keras.layers.Dense(TEXT_EMBEDDING_DIM)
        self.norm = keras.layers.LayerNormalization()

    def call(self, q, k, v, use_causal=False):
        Q = self.Q(q)
        K = self.K(k)
        V = self.V(v)
        splitted_Q = tf.split(Q, num_or_size_splits=ATTENTION_HEAD, axis=-1)
        splitted_K = tf.split(K, num_or_size_splits=ATTENTION_HEAD, axis=-1)
        splitted_V = tf.split(V, num_or_size_splits=ATTENTION_HEAD, axis=-1)
        tensors = [head(q, k, v, use_causal=use_causal) for head, q, k, v in zip(self.heads, splitted_Q, splitted_K, splitted_V)]
        return self.norm(tf.concat(tensors, axis=-1))


class MultiheadStaticAttention(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.Q = keras.layers.Dense(TEXT_EMBEDDING_DIM)
        self.V = keras.layers.Dense(TEXT_EMBEDDING_DIM)
        self.norm = keras.layers.LayerNormalization()

    def build(self, input_shape):
        self.memory = self.add_weight(
            shape=(EXPANSION_LENGTH, ATTENTION_HEAD, TEXT_EMBEDDING_DIM // ATTENTION_HEAD), # ehd
            initializer="glorot_uniform",
            trainable=True,
            name="memory"
        )

    def call(self, q, k, v):
        batch = tf.shape(q)[0]
        length = tf.shape(q)[2]
        dim = tf.shape(q)[3]
        Q_ = self.Q(q)
        V_ = self.V(k)
        Q = tf.reshape(Q_, (batch, length, ATTENTION_HEAD, dim // ATTENTION_HEAD)) # blhd
        V = tf.reshape(V_, (batch, length, ATTENTION_HEAD, dim // ATTENTION_HEAD)) # blhd
        eij = tf.einsum('ehd, blhd -> belh', self.memory, Q)
        eij = tf.nn.relu(eij) # belh
        out1 = tf.einsum('belh, blhd -> behd', eij, V) # behd
        out2 = tf.einsum('behd, belh -> blhd', out1, eij) # blhd
        out = self.norm(tf.reshape(out2, (batch, 1, length, dim)))
        return out


class DynamicAttention(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.Q = keras.layers.Dense(TEXT_EMBEDDING_DIM)
        self.K = keras.layers.Dense(TEXT_EMBEDDING_DIM)
        self.V = keras.layers.Dense(TEXT_EMBEDDING_DIM)

        self.memory = self.add_weight(
            shape=(1, EXPANSION_LENGTH, ATTENTION_HEAD, TEXT_EMBEDDING_DIM // ATTENTION_HEAD), # bnehd
            initializer="glorot_uniform",
            trainable=True,
            name="memory"
        )

    def call(self, q, k, v):
        Q = self.Q(q) # bnqd
        K = self.K(k)
        V = self.V(v)

        Q = tf.reshape(Q, (tf.shape(q)[0], tf.shape(q)[1], tf.shape(q)[2], ATTENTION_HEAD, TEXT_EMBEDDING_DIM // ATTENTION_HEAD)) # BNQHD
        K = tf.reshape(K, (tf.shape(k)[0], tf.shape(k)[2], ATTENTION_HEAD, TEXT_EMBEDDING_DIM // ATTENTION_HEAD)) # BKHD
        V = tf.reshape(V, (tf.shape(v)[0], tf.shape(v)[2], ATTENTION_HEAD, TEXT_EMBEDDING_DIM // ATTENTION_HEAD))

        broad_sum = self.memory[..., tf.newaxis, :, :] + K[..., tf.newaxis, :, :, :] # BEKHD
        matmul = tf.einsum("bekhd, bnqhd -> bnekqh", broad_sum, Q) # BNEKQH
        M = tf.nn.relu(matmul) # BNEKQH
        E_seq = tf.einsum("bnekqh, bkhd -> bnekhd", M, V) # BNEKHD
        out = tf.einsum("bnekhd, bnekqh -> bnqhd", E_seq, M) # BNQHD
        final = tf.reshape(out, tf.shape(q))
        return final


class CapturedMultiheadStaticAttention(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.Q = keras.layers.Dense(TEXT_EMBEDDING_DIM)
        self.V = keras.layers.Dense(TEXT_EMBEDDING_DIM)
        self.norm = keras.layers.LayerNormalization()
        self.self_attn = keras.layers.MultiHeadAttention(ATTENTION_HEAD, TEXT_EMBEDDING_DIM)

    def build(self, input_shape):
        self.memory = self.add_weight(
            shape=(EXPANSION_LENGTH, ATTENTION_HEAD, TEXT_EMBEDDING_DIM // ATTENTION_HEAD), # ehd
            initializer="glorot_uniform",
            trainable=True,
            name="memory"
        )

    def call(self, q, k, v):
        batch = tf.shape(q)[0]
        length = tf.shape(q)[2]
        dim = tf.shape(q)[3]
        Q_ = V_ = self.self_attn(q, k, v)
        Q = tf.reshape(Q_, (batch, length, ATTENTION_HEAD, dim // ATTENTION_HEAD)) # blhd
        V = tf.reshape(V_, (batch, length, ATTENTION_HEAD, dim // ATTENTION_HEAD)) # blhd
        eij = tf.einsum('ehd, blhd -> belh', self.memory, Q)
        eij = tf.nn.relu(eij) # belh
        out1 = tf.einsum('belh, blhd -> behd', eij, V) # behd
        out2 = tf.einsum('behd, belh -> blhd', out1, eij) # blhd
        out = self.norm(tf.reshape(out2, (batch, 1, length, dim)))
        return out

class SwinVision(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.swin = TFSwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    def call(self, image, **kwargs):
        processed = self.image_processor(image, return_tensors="tf")
        return self.swin(**processed).last_hidden_state


class InceptionVision(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.backbone = keras.applications.InceptionV3(include_top=False)
        self.backbone.trainable = False
        self.adapt = keras.layers.Dense(TEXT_EMBEDDING_DIM)

    def call(self, image, **kwargs):
        processed = keras.applications.inception_v3.preprocess_input(image)
        return self.adapt(self.backbone(processed))


class EfficientNetVision(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.backbone = keras.applications.EfficientNetB2(include_top=False)
        self.backbone.trainable = False

    def call(self, image, **kwargs):
        processed = keras.applications.efficientnet.preprocess_input(image)
        return self.backbone(processed)


class ResnetVision(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.backbone = keras.applications.ResNet50V2(include_top=False)
        self.backbone.trainable = False

    def call(self, image, **kwargs):
        processed = keras.applications.resnet_v2.preprocess_input(image)
        return self.backbone(processed)


class VGGVision(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.backbone = keras.applications.VGG16(include_top=False)
        self.backbone.trainable = False

    def call(self, image, **kwargs):
        processed = keras.applications.vgg16.preprocess_input(image)
        return self.backbone(processed)


class Resnet152Vision(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.backbone = keras.applications.ResNet152V2(include_top=False)
        self.backbone.trainable = False

    def call(self, image, **kwargs):
        processed = keras.applications.resnet_v2.preprocess_input(image)
        return self.backbone(processed)


class ShortVision(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(512, 8, strides=(8, 8))
        self.conv2 = keras.layers.Conv2D(1024, 4, strides=(4, 4))

    def call(self, image, **kwargs):
        image = keras.applications.resnet_v2.preprocess_input(image)
        return self.conv2(self.conv1(image))


class MobileVision(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.backbone = keras.applications.MobileNetV2(include_top=False)
        self.backbone.trainable = False

    def call(self, image, **kwargs):
        processed = keras.applications.mobilenet_v2.preprocess_input(image)
        return self.backbone(processed)


class MeshedEncoder(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.m_attn = {
            0: MultiheadMemorizedAttention,
            1: MultiheadRelativePositionalSelfAttention,
            2: MultiheadAttention,
            3: MultiheadStaticAttention,
            4: CapturedMultiheadStaticAttention
        }[ATTENTION_CHOICE]()
        self.f = keras.Sequential([
            keras.layers.Dense(TEXT_EMBEDDING_DIM, activation='relu'),
            keras.layers.Dense(TEXT_EMBEDDING_DIM)
        ])
        self.norm = keras.layers.LayerNormalization()

    def call(self, inp):
        z = self.norm(self.m_attn(inp, inp, inp) + inp)
        x = self.norm(self.f(z) + z)
        return x


class MeshedDecoder(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.sa = SelfAttention()
        self.ca = Attention()
        self.dense = keras.layers.Dense(TEXT_EMBEDDING_DIM, activation='sigmoid')
        self.norm = keras.layers.LayerNormalization()
        self.f = keras.Sequential([
            keras.layers.Dense(TEXT_EMBEDDING_DIM, activation='relu'),
            keras.layers.Dense(TEXT_EMBEDDING_DIM)
        ])

    def call(self, inp):
        src, tgts = inp
        sa = self.norm(self.sa(src))
        gated = tf.zeros(tf.shape(sa), dtype=tf.float32)
        for tgt in tgts:
            c = self.norm(self.ca(sa, tgt, tgt) + sa)
            alpha = self.dense(tf.concat([sa, c], axis=-1))
            feed = alpha * c
            gated += feed
        f = self.norm(self.f(gated) + gated)
        return self.norm(f)


class MultiLayerMeshed(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.enc = [MeshedEncoder() for _ in range(MESHED_DEPTH)]
        self.dec = [MeshedDecoder() for _ in range(MESHED_DEPTH)]

    def call(self, inp, training=True):
        src, tgt = inp
        srclst = [tgt]
        for block in self.enc:
            srclst.append(block(srclst[-1]))
        out = src
        for dec in self.dec:
            out = dec((out, srclst))
        return out


class NoMeshDecoder(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.sa = SelfAttention()
        self.ca = Attention()
        self.norm = keras.layers.LayerNormalization()
        self.f = keras.Sequential([
            keras.layers.Dense(TEXT_EMBEDDING_DIM, activation='relu'),
            keras.layers.Dense(TEXT_EMBEDDING_DIM)
        ])

    def call(self, inp):
        src, tgt = inp
        sa = self.norm(self.sa(src))
        c = self.norm(self.ca(sa, tgt, tgt) + sa)
        f = self.norm(self.f(c) + c)
        return self.norm(f)


class MultiLayerNoMesh(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.enc = [MeshedEncoder() for _ in range(MESHED_DEPTH)]
        self.dec = [NoMeshDecoder() for _ in range(MESHED_DEPTH)]

    def call(self, inp, training=True):
        txt, img = inp
        for block in self.enc:
            img = block(img)
        x = txt
        for dec in self.dec:
            x = dec((x, img))
        return x


class T5Decoder(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.model = TFAutoModel.from_pretrained('t5-base', from_pt=True)
        # self.model.trainable = False
        self.adaptor = keras.layers.Dense(TEXT_EMBEDDING_DIM)

    def call(self, inp, training=True):
        src, tgts = inp
        tgt = self.adaptor(tf.concat(tgts, axis=-1))
        outs = []
        if training:
            length = NUM_CAPTIONS
        else:
            length = 1
        for i in range(length):
            out = self.model.decoder(None, encoder_hidden_states=tf.squeeze(tgt, axis=1), inputs_embeds=src[:, i, ...]).last_hidden_state
            outs.append(out[:, tf.newaxis, ...])
        output = tf.concat(outs, axis=1)
        return output


class T5MultiLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.enc = [MeshedEncoder() for _ in range(MESHED_DEPTH)]
        self.dec = T5Decoder()

    def call(self, inp, training=True):
        src, tgt = inp
        srclst = [tgt]
        for block in self.enc:
            srclst.append(block(srclst[-1]))
        out = self.dec((src, srclst), training=training)
        return out


class DynamicDecoder(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.sa = SelfAttention()
        self.ca = DynamicAttention()
        self.norm = keras.layers.LayerNormalization()
        self.f = keras.Sequential([
            keras.layers.Dense(TEXT_EMBEDDING_DIM, activation='relu'),
            keras.layers.Dense(TEXT_EMBEDDING_DIM)
        ])

    def call(self, inp):
        src, tgt = inp
        sa = self.norm(self.sa(src))
        c = self.norm(self.ca(sa, tgt, tgt) + sa)
        f = self.norm(self.f(c) + c)
        return self.norm(f)


class MultiLayerDynamic(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.enc = [EncoderLayer() for _ in range(MESHED_DEPTH)]
        self.dec = [DynamicDecoder() for _ in range(MESHED_DEPTH)]

    def call(self, inp, training=True):
        src, tgt = inp
        for block in self.enc:
            tgt = block(tgt)
        out = src
        for dec in self.dec:
            out = dec((out, tgt))
        return out


class MeshedFastCaption(keras.Model):
    def __init__(self):
        super().__init__()
        self.vision = {
            0: Resnet152Vision,
            1: ResnetVision,
            2: VGGVision,
            3: EfficientNetVision,
            4: ShortVision,
            5: SwinVision,
            6: InceptionVision,
            7: MobileVision,
        }[BACKBONE_CHOICE]()
        self.decoder = {
            0: MultiLayerMeshed,
            1: T5MultiLayer,
            2: MultiLayerNoMesh,
            3: MultiLayerDynamic,
        }[DECODER_ATTENTION_CHOICE]()
        self.adapt = keras.layers.Dense(TEXT_EMBEDDING_DIM)
        self.dense = keras.layers.Dense(VOCAB_SIZE, activation='softmax')
        self.embedding = SeqEmbedding()

    def call(self, inputs, training=True):
        img, txt = inputs
        img = self.vision(img)
        img = self.adapt(img) # (batch, 8, 8, 768)
        img = tf.reshape(img, [tf.shape(img)[0], tf.shape(img)[1] * tf.shape(img)[2], tf.shape(img)[3]])
        seq = self.embedding(txt) # (b, p, length, dim)
        out = self.decoder((seq, img[:, tf.newaxis, ...]), training=training)
        return self.dense(out)

    def predict_step(self, data):
        (img, _), _ = data
        out = MeshedFastCaption.batch_generate_caption(self, img)
        return out

    @staticmethod
    @tf.function
    def generate_caption(model, img):
        img = model.vision(img)
        img = model.adapt(img)
        img = tf.reshape(img, [tf.shape(img)[0], tf.shape(img)[1] * tf.shape(img)[2], tf.shape(img)[3]])
        txt = tf.zeros((1, 1, INPUT_SEQ_LENGTH), dtype=tf.int32)
        return MeshedFastCaption._generate_from_seed(model, img, txt, tf.constant(0, dtype=tf.int32))

    @staticmethod
    def fast_generate_caption(model, img):
        img = model.vision(img)
        img = model.adapt(img)
        img = tf.reshape(img, [tf.shape(img)[0], tf.shape(img)[1] * tf.shape(img)[2], tf.shape(img)[3]])
        txt = tf.zeros((1, 1, 1), dtype=tf.int32)
        return MeshedFastCaption._fast_generate_from_seed(model, img, txt, tf.constant(0, dtype=tf.int32))

    @staticmethod
    @tf.function
    def _generate_from_seed(model, img, txt, index):
        while tf.math.logical_not(tf.math.logical_or(tf.math.reduce_any(tf.equal(txt, 2)), tf.math.reduce_all(tf.not_equal(txt, 0)))):
            seq = model.embedding(txt)
            out = model.decoder((seq, img[:, tf.newaxis]))
            prob = model.dense(out)
            new_text = tf.argmax(prob, axis=-1, output_type=tf.int32)
            valid = tf.cast(tf.range(tf.shape(txt)[2]) <= index, dtype=tf.int32)
            new_text = new_text * valid
            txt = tf.concat([tf.ones((1, 1, 1), dtype=tf.int32), new_text[:, :, :-1]], axis=2)
            index = index + 1
        return tf.concat([txt[:, :, 1:], tf.zeros((1, 1, 1), dtype=tf.int32)], axis=2)

    @staticmethod
    def _fast_generate_from_seed(model, img, txt, index):
        while tf.math.reduce_all(txt != 2) & (tf.shape(txt)[-2] < INPUT_SEQ_LENGTH):
            seq = model.embedding(txt)
            out = model.decoder((seq, img[:, tf.newaxis]))
            prob = model.dense(out)
            new_text = tf.argmax(prob, axis=-1, output_type=tf.int32)
            txt = tf.concat([tf.ones((1, 1, 1), dtype=tf.int32), new_text], axis=2)
            index = index + 1
        return txt[..., 1:]

    @staticmethod
    @tf.function
    def batch_generate_caption(model, imgs):
        imgs = model.vision(imgs)
        imgs = model.adapt(imgs)
        imgs = tf.reshape(imgs, [tf.shape(imgs)[0], tf.shape(imgs)[1] * tf.shape(imgs)[2], tf.shape(imgs)[3]])
        txt = tf.zeros((tf.shape(imgs)[0], 1, INPUT_SEQ_LENGTH), dtype=tf.int32)
        out = MeshedFastCaption._batch_generate_from_seed(model, imgs, txt, tf.constant(0, dtype=tf.int32))
        return out

    @staticmethod
    @tf.function
    def _batch_generate_from_seed(model, imgs, txt, index):
        finished_lines = tf.cast(tf.zeros((tf.shape(imgs)[0], 1), dtype=tf.int32), dtype=tf.bool)
        new_text = tf.zeros((tf.shape(imgs)[0], 1, MAXIMUM_LENGTH), dtype=tf.int32)
        while tf.math.logical_not(tf.reduce_all(finished_lines)) & (index <= tf.shape(txt)[2]):
            seq = model.embedding(txt)
            out = model.decoder((seq, imgs[:, tf.newaxis]), training=False)
            prob = model.dense(out)
            new_text = tf.argmax(prob, axis=-1, output_type=tf.int32)
            valid = tf.cast(tf.range(tf.shape(txt)[2]) <= index, dtype=tf.int32)
            new_text = new_text * valid
            proposed_txt = tf.concat([tf.ones((tf.shape(new_text)[0], 1, 1), dtype=tf.int32), new_text[..., :-1]], axis=-1)
            txt = tf.where(finished_lines[..., tf.newaxis], txt, proposed_txt)
            finished_lines = tf.reduce_any(txt == 2, axis=-1) # B, 1
            index = index + 1
        return tf.concat([txt[..., 1:], tf.zeros((tf.shape(new_text)[0], 1, 1), dtype=tf.int32)], axis=-1)


EXPERIMENTAL = MeshedFastCaption
EPOCH = 25
lr = LR = LearningRate = 1e-4

def get_init_epoch():
    return 0

def get_model_lastest():
    model = EXPERIMENTAL()
    img_inp = keras.Input((256, 256, 3))
    txt_inp = keras.Input((5, 70), dtype=tf.int32)
    model((img_inp, txt_inp))
    if highest > 0:
        model.load_weights(f'working.weights.h5')
    return model

def prepare_model():
    model = get_model_lastest()
    model.compile()
    return model

reverse_dict = {v: k for k, v in tokenize_dict.items()}
model = prepare_model()
def inference(image_array, printout=True):
    def num_to_str(value):
        if value in reverse_dict:
            return reverse_dict[value]
        return str(value)
    
    out_token = [EXPERIMENTAL.generate_caption(model, tf.constant(image_array)[tf.newaxis, ...]).numpy()[0][0]]
    label = label[tf.newaxis, ...]
    txtout = [' '.join(num_to_str(w) for w in ws).strip().split() for ws in out_token]
    txtreference = [[' '.join(filter(lambda x: len(x) > 0, (reverse_dict[int(w)] for w in ws))).strip().split() for ws in wws if len(set(ws)) > 1] for wws in label.numpy()]
    return inp, txtout

# --- Flask API (replace your previous Flask block with this) ---
from flask import Flask, request, jsonify
import io
import os
import traceback
from PIL import Image
import numpy as np

app = Flask(__name__)

# keep your inference function mapped here
INFER = inference  # ensure `inference` is defined above

def pil_image_to_numpy(img: Image.Image, target_size=None) -> np.ndarray:
    """Convert PIL image to uint8 numpy array (H,W,3). Optionally resize."""
    if target_size is not None:
        img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    img = img.convert('RGB')
    arr = np.array(img)
    return arr

@app.route('/predict', methods=['POST'])
def predict():
    # Basic sanity checks
    if INFER is None:
        return jsonify(error="No inference function found on server"), 500

    if 'file' not in request.files:
        return jsonify(error="No file part. Please upload with multipart form field named 'file'"), 400

    uploaded = request.files['file']
    if uploaded.filename == '':
        return jsonify(error="Empty filename"), 400

    try:
        # Read bytes and open with PIL (works for JPEG/PNG/etc)
        img_bytes = uploaded.read()
        img = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=f"Failed to read uploaded file as an image: {str(e)}"), 400

    try:
        # Convert to numpy array and resize to expected input shape
        # Use the IDEAL_SHAPE constant defined in your module (height, width, channels)
        target_size = IDEAL_SHAPE if 'IDEAL_SHAPE' in globals() else (256, 256, 3)
        img_array = pil_image_to_numpy(img, target_size=target_size)

        # If your model expects float inputs or special preprocessing, your
        # inference function should handle that (many of your model classes call preprocess_input).
        # Call INFER and accept flexible return types.
        result = INFER(img_array, printout=False)

        # Normalize return into JSON-friendly structure:
        # If inference returns a tuple/list/dict — try to jsonify as-is;
        # otherwise, wrap it.
        if isinstance(result, (dict, list, str, int, float, bool)):
            payload = result
        elif isinstance(result, tuple):
            # try to make tuple elements JSON serializable
            payload = {"result_tuple": [r.tolist() if hasattr(r, "tolist") else r for r in result]}
        else:
            # fallback: try converting to list/str
            try:
                payload = {"result": result.tolist()}
            except Exception:
                payload = {"result_str": str(result)}

        return jsonify({"status": "ok", "data": payload})

    except Exception as e:
        traceback.print_exc()
        return jsonify(error=f"Internal server error: {str(e)}"), 500

if __name__ == '__main__':
    # use 0.0.0.0 if you want external access (e.g., docker)
    app.run(host='0.0.0.0', port=8001)








