import tensorflow as tf
import numpy as np

def sigma(x, alpha=1):
  return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


class InfinyAttention(tf.keras.layers.Layer):
  def __init__(self, num_heads, d_model):
    super(InfinyAttention, self).__init__()

    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0
    self.depth = d_model // self.num_heads

    self.query_dense = tf.keras.layers.Dense(d_model)
    self.key_dense = tf.keras.layers.Dense(d_model)
    self.value_dense = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, inputs, batch_size):
    length = tf.shape(inputs)[1]
    inputs = tf.reshape(inputs, (batch_size, self.seq_count, self.seq_length, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 1, 4, 2, 3])

  def local_scaled_attention(self, q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
      scaled_attention_logits += (mask * 1e-9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output

  def infini_attention(self, query, key, value, mask, batch_size, delta_rule):
    local_attention_values = []
    sigma_query = sigma(query)
    sigma_key = sigma(key)
    Memory = tf.random.uniform((self.num_heads, self.num_heads), minval=0, maxval=0.1)
    z = tf.random.uniform((self.num_heads, ), minval=0, maxval=0.1)
    beta = tf.random.uniform((batch_size, self.depth, self.seq_length, self.num_heads), minval=0, maxval=0.1)
    for i in range(self.seq_count):
      local_sigma_query = sigma_query[:, i, :, :, :]
      local_sigma_key = sigma_key[:, i, :, :, :]

      A = tf.matmul(local_sigma_query, Memory) / (local_sigma_query * z)

      val = value[:, i, :, :, :]
      if delta_rule:
        val -= tf.matmul(local_sigma_key, Memory) / (local_sigma_key * z)

      Memory += tf.matmul(tf.transpose(local_sigma_key, perm=[0, 1, 3, 2]), val)
      z += tf.reduce_sum(local_sigma_key, axis=0)

      dot_attention = self.local_scaled_attention(query[:, i, :, :, :], key[:, i, :, :, :], value[:, i, :, :, :], mask)
      local_attention = tf.multiply(sigmoid(beta), A) + tf.multiply(1 - sigmoid(beta), dot_attention)
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

    attention_output = self.infini_attention(query, key, value, mask, batch_size, delta_rule)
    attention_output = tf.reshape(attention_output, (batch_size, self.d_model, length))
    attention_output = tf.transpose(attention_output, perm=[0, 2, 1])
    final_output = self.dense(attention_output)
    return final_output