# https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_tf_bert.py
import tensorflow as tf
import numpy as np
from utils import get_tensor_shape

class ModifiedBertEmbedding(tf.keras.layers.Layer):
    def __init(self, config, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range
        self.type_vocab_size = config.type_vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.layer_norm_eps = config.layer_norm_eps
        self.hidden_dropout_rate = config.hidden_dropout_rate
        
        initializer_func = tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)
        self.PositionEmbedding = tf.keras.layers.Embedding(
            self.max_position_embeddings,
            self.hidden_size,
            embeddings_initializer=initializer_func,
            name='position_embedding'
        )

        initializer_func = tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)
        self.TokenTypeEmbedding = tf.keras.layers.Embedding(
            self.type_vocab_size,
            self.hidden_size,
            embeddings_initializer=initializer_func,
            name='token_type_embedding'
        )

        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            name='layer_norm'
        )

        self.Dropout = tf.keras.layers.Dropout(self.hidden_dropout_rate)
    
    def build(self, input_shape):
        initializer_func = tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)
        self.word_embed_mapping = self.add_weight(
            name='word_embedding',
            shape=[self.vocab_size, self.hidden_size],
            initializer=initializer_func
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

class ModifiedBertAttention(tf.keras.layers.Layer):
    '''Performs multi-headed attention from `tensor_in` to `to_tensor`.

    '''
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        
        if config.hidden_size % config.num_attention_head != 0:
            raise ValueError(
                f'Hidden size {config.hidden_size} is not a multiple of'
                f'the number of attention heads {config.num_attention_head}'
            )
        
        self.hidden_size = config.hidden_size
        self.num_attention_head = config.num_attention_head
        self.unit_per_head = config.hidden_size / config.num_attention_head
        self.initializer_range = config.initializer_range
        self.attention_drop_rate = config.attention_drop_rate
        
        initializer_func = tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)
        self.QueryLayer = tf.keras.layers.experimental.EinsumDense(
            equation='abc,cde->abde',
            output_shape=(None, self.num_attention_head, self.unit_per_head),
            bias_axes='de',
            kernel_initializer=initializer_func,
            name='query'
        )
        
        initializer_func = tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)
        self.KeyLayer = tf.keras.layers.experimental.EinsumDense(
            equation='abc,cde->abde',
            output_shape=(None, self.num_attention_head, self.unit_per_head),
            bias_axes='de',
            kernel_initializer=initializer_func,
            name='key'
        )

        initializer_func = tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)
        self.ValueLayer = tf.keras.layers.experimental.EinsumDense(
            equation='abc,cde->abde',
            output_shape=(None, self.num_attention_head, self.unit_per_head),
            bias_axes='de',
            kernel_initializer=initializer_func,
            name='value'
        )

        self.Dropout = tf.keras.layers.Dropout(rate=self.attention_drop_rate)
    
    def call(
        self, 
        tensor_inputs, 
        attention_mask=None, 
        head_mask=None, 
        output_attention=False,
        training=False):
        '''Performs multi-head attention from `tensor_inputs`.

        The function first projects `tensor_inputs` into query, key, value 
        tensors. There are a list of tensors of length `num_attention_heads`,
        where each tensor's shape is [batch_size, seq_length, size_per_head]

        Then, the query and key tensors are dot-producted and scaled (if any).
        These are softmaxed to obtain attention probabilities. The value tensors
        are then interpolated by these probabilities, then concatenated back to
        a single tensor and returned.

        Args:
            tensor_inputs: list of float tensors
                If `tensor_inputs`'s length is 1 or tensors in `tensor_inputs` 
                are the same, then this is self-attention.
            attention_mask: (optional) float32 Tensor shape [batch_size, 
            query_seq_length, key_seq_length]
            head_mask: (optional) float32 Tensor.
            output_attention: boolean (Default: False).
                Whether the result additionally returns attention_probs.
            training: boolean (Default: False).
        Returns:
            float Tensor of shape [batch_size, query_seq_length, 
            num_attention_head, unit_per_head].
        Raises:
            ValueError: Any of the arguments or tensor shapes are invalid.
        '''
        # Scalar dimension reference:
        #   B = batch size
        #   Lq = sequence length of query tensor
        #   Lk = sequence length of key tensor
        #   Lv = sequence length of value tensor, It is equal to Lk
        #   H = number of attention heads
        #   U = number of units per head
        #   dim = dimension of inputs

        # [query, key, value] should be [B, Lq, dim], [B, Lk, dim], [B, Lv, dim]
        query_tensor = tensor_inputs[0]
        key_tensor = tensor_inputs[0] if len(tensor_inputs) == 1 else tensor_inputs[1]
        value_tensor = key_tensor if len(tensor_inputs) < 3 else tensor_inputs[2]

        # `query_tensor` = [B, Lq, H, U]
        # `key_tensor` = [B, Lk, H, U]
        # `value_tensor` = [B, Lv, H, U]
        query_tensor = self.QueryLayer(query_tensor)
        key_tensor = self.KeyLayer(key_tensor)
        value_tensor = self.ValueLayer(value_tensor)

        # attention score = QK^T/scale_factor = [B, H, Lq, Lk]
        scale_factor = 1/np.sqrt(self.unit_per_head)
        attention_scores = tf.einsum('aecd,acbd->acbe', key_tensor, query_tensor)
        attention_scores = tf.math.scalar_mul(scale_factor, attention_scores)

        if attention_mask is not None:
            # convert to size [B, 1, Lq, Lk]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])
            attention_scores = attention_scores + attention_mask
        
        # attention_probs = [B, H, Lq, Lk]
        attention_probs = tf.nn.softmax(attention_scores)
        attention_probs = self.Dropout(attention_probs)

        # Mask heads, however, in reference they calculate on attention_scores.
        # Should it be attention_probs? For now, we comment this.
        # if head_mask is not None:
        #     attention_probs = attention_probs * head_mask

        # `attention_output` = [B, Lq, H, U]
        attention_output = tf.einsum('acbe,aecd->abcd', attention_probs, value_tensor)
        output = (attention_output,)
        if output_attention:
            output = (attention_output, attention_probs)
        return output
