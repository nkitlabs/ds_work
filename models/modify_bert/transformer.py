# https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_tf_bert.py
import tensorflow as tf
import numpy as np
from .utils import get_tensor_shape
from .activate_function import gelu_activate_fn
from .result_template import ResultBertEncoder, ResultBertMainLayer

class ModifiedBertEmbedding(tf.keras.layers.Layer):
    def __init__(
        self, 
        output_dim,
        vocab_size,
        type_vocab_size,
        max_position_embeddings,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.2),
        layer_norm_eps=1e-12,
        drop_rate=0.2,
        **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.kernel_initializer = kernel_initializer
        self.type_vocab_size = type_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.drop_rate = drop_rate
        
        self.PositionEmbedding = tf.keras.layers.Embedding(
            self.max_position_embeddings,
            self.output_dim,
            embeddings_initializer=self.kernel_initializer,
            name='position_embedding'
        )

        self.TokenTypeEmbedding = tf.keras.layers.Embedding(
            self.type_vocab_size,
            self.output_dim,
            embeddings_initializer=self.kernel_initializer,
            name='token_type_embedding'
        )

        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            name='layer_norm'
        )

        self.Dropout = tf.keras.layers.Dropout(self.drop_rate)
    
    def build(self, input_shape):
        self.word_embed_mapping = self.add_weight(
            name='word_embedding',
            shape=[self.vocab_size, self.output_dim],
            initializer=self.kernel_initializer
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
                token_type_ids,
                input_embeds, 
                training=training)
        elif mode == 'linear':
            return self._linear(input_ids)
        else:
            raise ValueError(f'mode {mode} is invalid')
    
    def _embedding(
        self, 
        input_ids, 
        position_ids, 
        input_embeds, 
        token_type_ids,
        training=False):
        '''Applies embedding based on inputs tensor.'''
        assert not (input_ids is None and input_embeds is None)

        # create different types of embeds.
        _input = input_ids if input_ids is not None else input_embeds
        input_shape = get_tensor_shape(_input)
        if input_embeds is None:
            input_embeds = tf.gather(
                self.word_embed_mapping, 
                tf.cast(input_ids, dtype=tf.int32)
            )

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

class ModifiedBertAttention(tf.keras.layers.Layer):
    '''Performs multi-headed attention from tensor_inputs.

    The object consists of three EinsumDense Layers to support query (Q), key (K), 
    and value (V) tensors, Dropout Layers, Linear and a normalization layer
    as described in the paper `attention is all you need`.

    Args:
        units: Positive integer. Total number of output space.
        num_head: Positive integer. number of attention heads in this layer.
        kernel_initializer: Initializer for the weight matrix.
        drop_rate: Float between 0 and 1. Fraction of the input units to drop
            in a Dropout layer.
        layer_norm_eps: Small float added to variance to avoid dividing by zero.
        is_scale: Boolean. Whether the attention score should be scaled.
    '''
    def __init__(
        self, 
        units,
        num_head=1,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.2),
        drop_rate=0.2,
        layer_norm_eps=1e-12,
        is_scale=True, 
        **kwargs):
        super().__init__(**kwargs)
        
        if units % num_head != 0:
            raise ValueError(
                f'the Number of units {units} is not a multiple of'
                f'the number of attention heads {num_head}'
            )
        
        self.units = units
        self.num_head = num_head
        self.unit_per_head = units // num_head
        self.kernel_initializer = kernel_initializer
        self.drop_rate = drop_rate
        self.layer_norm_eps = layer_norm_eps
        self.is_scale = is_scale
        
        self.QueryLayer = tf.keras.layers.experimental.EinsumDense(
            equation='abc,cde->abde',
            output_shape=(None, self.num_head, self.unit_per_head),
            bias_axes='de',
            kernel_initializer=self.kernel_initializer,
            name='query'
        )
        
        self.KeyLayer = tf.keras.layers.experimental.EinsumDense(
            equation='abc,cde->abde',
            output_shape=(None, self.num_head, self.unit_per_head),
            bias_axes='de',
            kernel_initializer=self.kernel_initializer,
            name='key'
        )

        self.ValueLayer = tf.keras.layers.experimental.EinsumDense(
            equation='abc,cde->abde',
            output_shape=(None, self.num_head, self.unit_per_head),
            bias_axes='de',
            kernel_initializer=self.kernel_initializer,
            name='value'
        )

        self.Dropout1 = tf.keras.layers.Dropout(rate=self.drop_rate)

        self.Dense = tf.keras.layers.experimental.EinsumDense(
            equation='abcd,cde->abe',
            output_shape=(None, self.units),
            bias_axes='e',
            kernel_initializer=self.kernel_initializer,
            name='dense',
        )
        
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            name='LayerNorm',
        )
        self.Dropout2 = tf.keras.layers.Dropout(rate=self.drop_rate)
    
    def call(self, tensor_inputs, tensor_mask=None, training=False):
        '''Performs multi-head attention from `tensor_inputs`.

        The function first projects `tensor_inputs` into query, key, value 
        tensors. There are a list of tensors of length `num_attention_heads`,
        where each tensor's shape is [batch_size, seq_length, units]

        Then, the query and key tensors are dot-producted and scaled (if any).
        These are softmaxed to obtain attention probabilities. The value tensors
        are then interpolated by these probabilities. Finally, the results and
        the input is passed into a residual and a normalized layer.

        Args:
            tensor_inputs: list of float tensors [Query, Key, Value] or float tensor.
                If `tensor_inputs`'s length is 1 or it is a float tensor,
                are the same, then this is self-attention.
            tensor_mask: (optional) float32 Tensor shape [batch_size, 
            query_seq_length, key_seq_length]
                A mask whether a pair of items in query and key should be 
                calculated or not (one means that pair should be calculated).
            training: boolean (Default: False).
        Returns:
            - float Tensor of shape [batch_size, query_seq_length, 
                num_attention_head * unit_per_head (dimension input)].
            - (conditional: if does_return_attention_probs is True)
                float Tensor of `attention_probs` [batch_size, number_of_head, 
                query_seq_length, key_seq_length]
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
        #   dim = dimension of inputs, It's equal to `units` (H*U)

        # [query, key, value] should be [B, Lq, dim], [B, Lk, dim], [B, Lv, dim]
        tensor_query = None
        tensor_key = None
        tensor_value = None

        if isinstance(tensor_inputs, list):
            tensor_query = tensor_inputs[0]
            tensor_key = tensor_inputs[0] if len(tensor_inputs) == 1 else tensor_inputs[1]
            tensor_value = tensor_key if len(tensor_inputs) < 3 else tensor_inputs[2]
        elif tf.is_tensor(tensor_inputs):
            tensor_query = tensor_inputs
            tensor_key = tensor_inputs
            tensor_value = tensor_inputs
        else:
            raise ValueError(
                f'`tensor_inputs` should be either list of float tensors or'
                f'a float tensor'
            )

        # attention_probs = [B, H, Lq, Lk]
        attention_probs = self.cal_attention_probs(
            tensor_query,
            tensor_key,
            tensor_mask,
        )
        attention_probs = self.Dropout1(attention_probs, training=training)

        # `tensor_value` = [B, Lv, H, U]
        # `tensor_out` = [B, Lq, H, U]
        value_tensor = self.ValueLayer(tensor_value)
        tensor_out = tf.einsum('acbe,aecd->abcd', attention_probs, value_tensor)
        
        # `attention_output` = [B, Lq, H*U (dim)]
        tensor_out = self.Dense(tensor_out)
        tensor_out = self.Dropout2(tensor_out, training=training)
        tensor_out = self.LayerNorm(tensor_out + tensor_inputs[0])
        return tensor_out
    
    def cal_attention_probs(self, tensor_query, tensor_key, tensor_mask=None):
        '''Calculate attention probabilities from query and key tensor.
        Args:
            tensor_query: A float tensor whose size [batch_size, length, dim_input]
            tensor_key: A float tensor whose size [batch_size, length, dim_input]
            tensor_mask: A float tensor whose size [batch_size, query_length, 
            key_length]
                A mask whether a pair of items in query and key should be 
                calculated or not.
        Returns:
            - float Tensor of shape [batch_size, query_seq_length, 
                num_attention_head * unit_per_head (dimension input)].
        '''
        # `tensor_query` => [B, Lq, H, U]
        # `tensor_key` => [B, Lk, H, U]
        tensor_query = self.QueryLayer(tensor_query)
        tensor_key = self.KeyLayer(tensor_key)

        # attention score = QK^T/scale_factor => [B, H, Lq, Lk]
        attention_scores = tf.einsum('aecd,abcd->acbe', tensor_key, tensor_query)
        if self.is_scale:
            scale_factor = 1/np.sqrt(self.unit_per_head)
            attention_scores = tf.math.scalar_mul(scale_factor, attention_scores)

        if tensor_mask is not None:
            # convert `mask_tensor` => [B, 1, Lq, Lk]
            tensor_mask = tf.expand_dims(tensor_mask, axis=[1])
            attention_scores = attention_scores + tensor_mask
        
        attention_probs = tf.nn.softmax(attention_scores)
        return attention_probs

class ModifiedBertSubLayer(tf.keras.layers.Layer):
    '''Performs BERT unit.

    The object consists of an attention layer in sequence with a feed-forward 
    network and a normalize layer as described in the paper 
    `attention is all you need`.

    Args:
        units: Positive integer. Total number of output space.
        feed_forward_size: Positive integer. The number of units in 
            a feed-froward layer.
        num_head: Positive integer. number of attention heads in this layer.
        kernel_initializer: Initializer for the weight matrix.
        activation_fn: Function or string. the representative of activation
        function in a feed-forward layer.
        drop_rate: Float between 0 and 1. Fraction of the input units to drop
            in a Dropout layer.
        layer_norm_eps: Small float added to variance to avoid dividing by zero.
        is_scale: Boolean. Whether the attention score should be scaled.
    '''
    def __init__(
        self, 
        units,
        feed_forward_size=3072,
        num_head=1,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.2),
        activation_fn=gelu_activate_fn,
        drop_rate=0.2,
        layer_norm_eps=1e-12,
        is_scale=True, 
        **kwargs):
        
        super().__init__(**kwargs)
        self.units = units
        self.feed_forward_size = feed_forward_size
        self.num_head = num_head
        self.kernel_initializer = kernel_initializer
        self.activation_fn = activation_fn
        self.drop_rate = drop_rate
        self.layer_norm_eps = layer_norm_eps
        self.is_scale = is_scale

        self.Attention = ModifiedBertAttention( 
            units,
            num_head=num_head,
            kernel_initializer=kernel_initializer,
            drop_rate=drop_rate,
            layer_norm_eps=layer_norm_eps,
            is_scale=is_scale, 
            name='attention',
        )

        self.FeedForward = tf.keras.layers.experimental.EinsumDense(
            equation='abc,cd->abd',
            output_shape=(None, feed_forward_size),
            bias_axes='d',
            kernel_initializer=kernel_initializer,
            name='feed_forward',
        )
        self.FeedForwardActFn = tf.keras.layers.Activation(activation_fn)

        self.Dense = tf.keras.layers.experimental.EinsumDense(
            equation='abc,cd->abd',
            bias_axes='d',
            output_shape=(None, units),
            kernel_initializer=kernel_initializer,
            name='dense',
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            name='LayerNorm',
        )
        self.Dropout = tf.keras.layers.Dropout(rate=self.drop_rate)
    
    def call(
        self,
        tensor_inputs, 
        tensor_mask=None, 
        training=False):
        '''
        Args:
            tensor_inputs: list of float tensors [Query, Key, Value]
                If `tensor_inputs`'s length is 1 or tensors in `tensor_inputs` 
                are the same, then this is self-attention.
            tensor_mask: (optional) float32 Tensor shape [batch_size, 
            query_seq_length, key_seq_length]
            training: boolean (Default: False).
        Returns:
            - float Tensor of shape [batch_size, query_seq_length, 
                num_attention_head * unit_per_head (dimension input)].
            - (conditional: if does_return_attention_probs is True)
                float Tensor of `attention_probs` [batch_size, number_of_head, 
                query_seq_length, key_seq_length]
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
        #   dim = dimension of inputs, It's equal to `units` (H*U)
        #   F = number of units in a feed-forward layer

        # `attention_output` = [B, Lq, H*U (dim)]
        attention_output = self.Attention(tensor_inputs, tensor_mask, training)
        
        # `feed_forward_output` = [B, Lq, F]
        feed_forward_output = self.FeedForward(attention_output)
        feed_forward_output = self.FeedForwardActFn(feed_forward_output)
        
        # `attention_output` = [B, Lq, dim]
        tensor_out = self.Dense(feed_forward_output)
        tensor_out = self.Dropout(tensor_out, training=training)
        tensor_out = self.LayerNorm(tensor_out + attention_output)
        
        return tensor_out
    
    def cal_attention_probs(self, tensor_query, tensor_key, tensor_mask=None):
        return self.Attention.cal_attention_probs(tensor_query, tensor_key, tensor_mask)

class ModifiedBertEncoder(tf.keras.layers.Layer):
    def __init__(
        self, 
        units,
        feed_forward_size=3072,
        num_head=1,
        num_hidden_layers=12,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.2),
        activation_fn=gelu_activate_fn,
        drop_rate=0.2,
        layer_norm_eps=1e-12,
        is_scale=True, 
        **kwargs):
        
        super().__init__(**kwargs)
        self.units = units
        self.feed_forward_size = feed_forward_size
        self.num_head = num_head
        self.num_hidden_layers = num_hidden_layers
        self.kernel_initializer = kernel_initializer
        self.activation_fn = activation_fn
        self.drop_rate = drop_rate
        self.layer_norm_eps = layer_norm_eps
        self.is_scale = is_scale

        self.Layer = [
            ModifiedBertSubLayer(
                units,
                feed_forward_size=feed_forward_size,
                num_head=num_head,
                kernel_initializer=kernel_initializer,
                activation_fn=activation_fn,
                drop_rate=drop_rate,
                layer_norm_eps=layer_norm_eps,
                is_scale=is_scale, 
                name="layer_._{}".format(i)
            ) 
            for i in range(num_hidden_layers)
        ]
    
    def cal_hidden_states(self, tensor_in, mask_tensor=None, training=False):
        '''calculate hidden states from sequential BertSubLayers.
        
        Args:
            tensor_inputs: list of float tensors [Query, Key, Value]
                If `tensor_inputs`'s length is 1 or tensors in `tensor_inputs` 
                are the same, then this is self-attention.
            tensor_mask: (optional) float32 Tensor shape [batch_size, 
            query_seq_length, key_seq_length]
            training: boolean (Default: False).
        Returns:
            all_hidden_states: list of float tensors
        '''
        all_hidden_states = (tensor_in, )
        _tensor = tensor_in
        for _bert_layer in self.Layer:
            # self-attention
            _output = _bert_layer(
                [_tensor],
                mask_tensor=mask_tensor,
                training=training
            )
            _tensor = _output
            all_hidden_states += _tensor
        return all_hidden_states
    
    def get_calculated_value(self, tensor_in, mask_tensor=None, training=False):
        '''Return all hidden states and attention probs calculated in the layer.
        
        Args:
            tensor_inputs: list of float tensors [Query, Key, Value]
                If `tensor_inputs`'s length is 1 or tensors in `tensor_inputs` 
                are the same, then this is self-attention.
            tensor_mask: (optional) float32 Tensor shape [batch_size, 
            query_seq_length, key_seq_length]
            training: boolean (Default: False).
        Returns:
            ResultBertEncoder object 
                consisting of last hidden state (output), all other hidden states,
                and attention probabilities.
        '''
        all_hidden_states = self.cal_hidden_states(
            tensor_in, 
            mask_tensor=mask_tensor, 
            training=training,
        )
        attention_probs = ()
        for i, _hidden_state in enumerate(all_hidden_states[:-1]):
            attention_probs += self.Layer[i].cal_attention_probs(
                 _hidden_state
                , _hidden_state
                , mask_tensor
            )
        return ResultBertEncoder(
            output=all_hidden_states[-1],
            hidden_states=all_hidden_states[:-1], 
            attentions=attention_probs,
        )

    def call(self, tensor_in, mask_tensor=None, training=False):
        all_hidden_states = self.cal_hidden_states(
            tensor_in, 
            mask_tensor=mask_tensor, 
            training=training,
        )
        tensor_out = all_hidden_states[-1]
        return tensor_out

class ModifiedBertMainLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        vocab_size,
        type_vocab_size,
        max_position_embeddings,
        feed_forward_size=3072,
        num_head=1,
        num_hidden_layers=12,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.2),
        activation_fn=gelu_activate_fn,
        pooling_activation='tanh',
        layer_norm_eps=1e-12,
        drop_rate=0.2,
        is_scale=True,
        has_pooling_layer=True,
        **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.feed_forward_size = feed_forward_size
        self.num_head = num_head
        self.num_hidden_layers = num_hidden_layers
        self.kernel_initializer = kernel_initializer
        self.activation_fn = activation_fn
        self.pooling_activation = pooling_activation
        self.layer_norm_eps = layer_norm_eps
        self.drop_rate = drop_rate
        self.is_scale = is_scale
        self.has_pooling_layer = has_pooling_layer

        self.Embedding = ModifiedBertEmbedding(
            self.units,
            self.vocab_size,
            self.type_vocab_size,
            self.max_position_embeddings,
            kernel_initializer=self.kernel_initializer,
            layer_norm_eps=self.layer_norm_eps,
            drop_rate=self.drop_rate,
            name='embedding'
        )
        
        self.Encoder = ModifiedBertEncoder(
            self.units,
            feed_forward_size=self.feed_forward_size,
            num_head=self.num_head,
            num_hidden_layers=self.num_hidden_layers,
            kernel_initializer=self.kernel_initializer,
            activation_fn=self.activation_fn,
            drop_rate=self.drop_rate,
            layer_norm_eps=self.layer_norm_eps,
            is_scale=self.is_scale, 
        )

        self.Pooling = tf.keras.layers.Dense(
            self.units,
            kernel_initializer=self.kernel_initializer,
            activation=self.pooling_activation,
            name='pooling',
        )

    def call(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        input_embeds=None,
        tensor_mask=None,
        training=False):
        _input = input_ids if input_ids is not None else input_embeds
        input_shape = get_tensor_shape(_input)
        if tensor_mask is None:
            tensor_mask = tf.fill(input_shape, 1)
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)
        
        embedding_output = self.Embedding(
            input_ids=input_ids,
            position_ids=position_ids, 
            token_type_ids=token_type_ids,
            input_embeds=input_embeds,
            mode='embedding',
            training=training
        )

        # we convert 2D 0,1 attention mask to 3D 0,-100000 attention mask with 
        # size [batch_size, 1, 1, key_length] so that we can broad cast to 
        # [batch_size, num_head, query_length, key_length]
        extended_attention_mask = tensor_mask[:, tf.newaxis, tf.newaxis, :]
        extended_attention_mask = tf.cast(extended_attention_mask, embedding_output.dtype)
        extended_attention_mask = -10000.0 * (1.0-extended_attention_mask)

        encoder_output = self.Encoder(
            embedding_output,
            tensor_mask=extended_attention_mask, 
            training=training,
        )
        tensor_out = encoder_output

        pooling_output = None
        if self.has_pooling_layer:
            pooling_output = self.Pooling(encoder_output)
            tensor_out = pooling_output
        
        return tensor_out
        # if not does_return_dict:
        #     return (
        #         encoder_outputs[0],
        #         pooling_output,
        #     ) + encoder_outputs[1:]
        
        # return ResultBertMainLayer(
        #     output=encoder_outputs[0],
        #     pooler_output=pooling_output,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        # )