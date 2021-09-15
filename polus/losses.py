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


def weighted_sigmoid_cross_entropy_from_logits(class_weights, negative_weight):
    """
    FUTURE_TODO: the weight can be directly added to the sigmoid cross entropy, probably it would make more sense
    """
    @tf.function
    def weighted_sigmoid_cross_entropy_from_logits_loss(y_true, y_pred):
        
        mask = tf.math.reduce_all(y_true == 0, axis=-1)
        negative_weight_per_sample = tf.cast(mask, tf.float32) * negative_weight
        
        # TODO add epsilon value so that weights never be zero

        weights = tf.reduce_sum(tf.constant(class_weights, dtype=tf.float32) * y_true, axis=-1) + negative_weight_per_sample
 

        unweighted_losses = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred), axis=-1)

        weighted_losses = unweighted_losses * weights

        return tf.reduce_mean(weighted_losses)
    #return tf.reduce_sum(comulative_error)/tf.size(comulative_error, out_type=tf.float32)
    return weighted_sigmoid_cross_entropy_from_logits_loss