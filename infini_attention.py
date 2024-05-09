import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def sigma(x, alpha=1):
  return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


class InfiniAttention(tf.keras.layers.Layer):
  def __init__(self, num_heads, d_model):
    super(InfiniAttention, self).__init__()

    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0
    self.depth = d_model // self.num_heads

    self.query_dense = tf.keras.layers.Dense(d_model)
    self.key_dense = tf.keras.layers.Dense(d_model)
    self.value_dense = tf.keras.layers.Dense(d_model)
    self.beta = tf.Variable(initial_value=1.0, trainable=True)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, inputs, batch_size):
    length = tf.shape(inputs)[1]
    inputs = tf.reshape(inputs, (batch_size, self.seq_count, self.seq_length, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 3, 1, 2, 4])

  def local_scaled_attention(self, q, k, v, local_seq_length, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask:
      mask = 1 - tf.linalg.band_part(tf.ones((local_seq_length, local_seq_length)), -1, 0)
      scaled_attention_logits += (mask * 1e-9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output

  def infini_attention(self, query, key, value, mask=False, delta_rule=False):
    local_attention_values = []
    batch_size, _, _, local_seq_length, _ = tf.shape(query)
    sigma_query = sigma(query)
    sigma_key = sigma(key)
    Memory = tf.zeros((self.num_heads, self.depth, self.depth))
    z = tf.zeros((self.num_heads, self.depth, 1))
    for i in range(self.seq_count):
      local_sigma_query = sigma_query[:, :, i, :, :]
      local_sigma_key = sigma_key[:, :, i, :, :]


      A = tf.matmul(local_sigma_query, Memory) / (tf.matmul(local_sigma_query, z) + 1e-6)


      val = value[:, :, i, :, :]

      if delta_rule:
        val -= tf.matmul(local_sigma_key, Memory) / (tf.matmul(local_sigma_key, z) + 1e-6)

      Memory += tf.matmul(tf.transpose(local_sigma_key, perm=[0, 1, 3, 2]), val)
      z += tf.reduce_sum(local_sigma_key, axis=-2, keepdims=True)

      dot_attention = self.local_scaled_attention(query[:, :, i, :, :], key[:, :, i, :, :], value[:, :, i, :, :], local_seq_length, mask)
      local_attention = sigmoid(self.beta) * A + (1 - sigmoid(self.beta)) * dot_attention
      local_attention_values.append(local_attention)

    O = tf.concat(local_attention_values, axis=1)
    return O

  def call(self, inputs, seq_count, delta_rule=True):
    query, key, value, mask = inputs
    batch_size = tf.shape(query)[0]
    length = tf.shape(query)[1]
    self.seq_count = seq_count
    self.seq_length = length // seq_count

    query = self.query_dense(query)
    key = self.key_dense(key)
    value = self.value_dense(value)

    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    attention_output = self.infini_attention(query, key, value, mask, delta_rule)
    attention_output = tf.reshape(attention_output, (batch_size, self.d_model, length))
    attention_output = tf.transpose(attention_output, perm=[0, 2, 1])
    final_output = self.dense(attention_output)
    return final_output