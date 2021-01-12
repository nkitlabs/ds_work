import tensorflow as tf
from activate_function import gelu_activate_fn
from transformer import ModifiedBertMainLayer
from result_template import ResultBertPretrainingMLM
from utils import get_tensor_shape

class MLMPredictionLayer(tf.keras.layers.Layer):
    def __init__(
        self, 
        units,
        vocab_size,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.2),
        activation_fn=gelu_activate_fn,
        layer_norm_eps=1e-12,
        **kwargs):
        
        self.vocab_size = vocab_size
        self.Dense = tf.keras.layers.Dense(
            units,
            kernel_initializer=kernel_initializer,
            name='dense',
        )
        self.Activation = tf.keras.layers.Activation(activation_fn)
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps,
            name='layer_norm',
        )

    def call(self, tensor_in):
        _tensor = self.Dense(tensor_in)
        _tensor = self.Activation(_tensor)
        _tensor = self.LayerNorm(_tensor)
        return _tensor

class PretrainingMLM(tf.keras.layers.Layer):
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
        **kwargs):
        super().__init__(**kwargs)

        self.Bert = ModifiedBertMainLayer(
            units,
            vocab_size,
            type_vocab_size,
            max_position_embeddings,
            feed_forward_size=feed_forward_size,
            num_head=num_head,
            num_hidden_layers=num_hidden_layers,
            kernel_initializer=kernel_initializer,
            activation_fn=activation_fn,
            pooling_activation=pooling_activation,
            layer_norm_eps=layer_norm_eps,
            drop_rate=drop_rate,
            is_scale=is_scale,
            has_pooling_layer=False,
            name='BERT'
        )
        
        self.MLM = MLMPredictionLayer(
            units,
            vocab_size,
            kernel_initializer=kernel_initializer,
            activation_fn=activation_fn,
            layer_norm_eps=layer_norm_eps,
            name='mlm_cls',
        )
    
    def compute_loss(self, labels, preds):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        # calculate only labels that not equal to -100
        active_loss = tf.not_equal(tf.reshape(labels, (-1,)), -100)
        preds = tf.boolean_mask(
            tf.reshape(preds, (-1, get_tensor_shape(preds)[2])),
            active_loss
        )
        labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
        return loss_fn(labels, preds)

    def call(
        self,
        input_ids=None,
        position_ids=None, 
        token_type_ids=None,
        input_embeds=None,
        labels=None,
        attention_mask=None,
        head_mask=None,
        does_return_attention_probs=False,
        does_return_hidden_state=False,
        training=False,
        does_return_dict=True):

        bert_outputs = self.Bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            head_mask=head_mask,
            does_return_attention_probs=does_return_attention_probs,
            does_return_hidden_state=does_return_hidden_state,
            training=training,
            does_return_dict=does_return_dict
        )
        preds = self.MLM(bert_outputs[0])
        loss = None
        if labels is not None:
            logits = preds[:, :-1]
            loss = self.compute_loss(labels, logits)

        
        if not does_return_dict:
            res = (preds, ) + bert_outputs[2:]
            if loss is not None:
                res = (loss,) + res
            return res
        
        return ResultBertPretrainingMLM(
            loss=loss,
            output=preds,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
        )
    