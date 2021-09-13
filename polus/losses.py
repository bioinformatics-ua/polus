import tensorflow as tf

## Definition of some losses functions

def weighted_softmax_cross_entropy_from_logits(class_weights):
        
    @tf.function
    def weighted_softmax_cross_entropy_from_logits_loss(y_true, y_pred):

        weights = tf.reduce_sum(tf.constant(class_weights, dtype=tf.float32) * y_true, axis=-1)

        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        weighted_losses = unweighted_losses * weights

        return tf.reduce_mean(weighted_losses)
    #return tf.reduce_sum(comulative_error)/tf.size(comulative_error, out_type=tf.float32)
    return weighted_softmax_cross_entropy_from_logits_loss


def weighted_sigmoid_cross_entropy_from_logits(class_weights):
        
    @tf.function
    def weightedd_sigmoid_cross_entropy_from_logits_loss(y_true, y_pred):


        weights = tf.reduce_sum(tf.constant(class_weights, dtype=tf.float32) * y_true, axis=-1)

        unweighted_losses = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=mock_labels, logits=mock_logits), axis=-1)

        weighted_losses = unweighted_losses * weights

        return tf.reduce_mean(weighted_losses)
    #return tf.reduce_sum(comulative_error)/tf.size(comulative_error, out_type=tf.float32)
    return weightedd_sigmoid_cross_entropy_from_logits_loss