import tensorflow as tf

def vector_equals(a, b, epsilon=0.00001):
    cos_d = tf.einsum("bse,bse->bs", a, b) / tf.einsum("bs,bs->bs", tf.norm(a, axis=-1, ord=2), tf.norm(b, axis=-1, ord=2)) 
    return tf.reduce_all(cos_d - 1 < epsilon)