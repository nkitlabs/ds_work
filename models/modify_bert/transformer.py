# https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_tf_bert.py
import tensorflow as tf
from utils import get_tensor_shape

class ModifiedBertEmbedding(tf.keras.layers.Layer):
    def __init(self, config):
        super().__init__()

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range
        self.type_vocab_size = config.type_vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.layer_norm_eps = config.layer_norm_eps
        self.hidden_dropout_rate = config.hidden_dropout_rate
        
        self.PositionEmbedding = tf.keras.layers.Embedding(
            self.max_position_embeddings,
            self.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(self.initializer_range),
            name='position_embedding'
        )

        self.TokenTypeEmbedding = tf.keras.layers.Embedding(
            self.type_vocab_size,
            self.hidden_size,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(self.initializer_range),
            name='token_type_embedding'
        )

        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            name='layer_norm'
        )

        self.Dropout = tf.keras.layers.Dropout(self.hidden_dropout_rate)
    
    def build(self, input_shape):
        self.word_embed_mapping = self.add_weight(
            name='word_embedding',
            shape=[self.vocab_size, self.hidden_size],
            initializer=tf.keras.initializers.TruncatedNormal(self.initializer_range)
        )
        # super().build(input_shape)

    def call(
        self, 
        input_ids=None,
        position_ids=None, 
        token_type_ids=None,
        input_embeds=None,
        mode='embedding',
        training=False):
        if mode == 'embedding':
            return self._embedding(
                input_ids, 
                position_ids, 
                input_embeds, 
                training=training)
        else:
            raise ValueError(f'mode {mode} is invalid')
    
    def _embedding(self, input_ids, position_ids, input_embeds, training=False):
        '''Applies embedding based on inputs tensor.'''
        assert not (input_ids is None and input_embeds is None)

        # create different types of embeds.
        _input = input_ids if input_ids is not None else input_embeds
        input_shape = get_tensor_shape(_input)
        if input_embeds is None:
            input_embeds = tf.gather(self.word_embed_mapping, input_ids)

        if position_ids is None:
            position_ids = tf.range(input_shape[1], dtype=tf.int32)[tf.newaxis, :]
        position_embeds = tf.cast(self.PositionEmbedding(position_ids), input_embeds.dtype)

        if token_type_ids is None:
            token_type_ids = tf.zeros(input_shape, dtype=tf.int32)
        token_type_embeds = tf.cast(self.TokenTypeEmbedding(token_type_ids), input_embeds.dtype)

        # concatenate and process embedding
        embeds = input_embeds + position_embeds + token_type_embeds
        embeds = self.LayerNorm(embeds)
        embeds = self.Dropout(embeds, training=training)
        return embeds
    
    # tensor input [batch_size, length, hidden_size]
    def linear(self, tensor):
        '''Computes logits by running inputs through a linear layer.

        Args:
            tensor: A float32 tensor with shape [batch_size, length, hidden_size]
        Returns:
            float32 tensor with shape [batch_size, length, vocab_size]

        '''
        input_shape = get_tensor_shape(tensor)
        tensor = tf.reshape(tensor, [-1, self.hidden_size])
        tensor = tf.matmul(tensor, self.word_embed_mapping, transpose_b=True)
        return tf.reshape(tensor, [input_shape[0], input_shape[1], self.vocab_size])



