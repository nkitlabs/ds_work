import tensorflow as tf

def get_tensor_shape(tensor):
    dynamic = tf.shape(tensor)
    if tensor.shape == tf.TensorShape(None):
        return dynamic
    
    static = tensor.shape.as_list()
    return [x if x is not None else dynamic[i] for i,x in enumerate(static)]