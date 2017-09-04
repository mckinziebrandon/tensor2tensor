# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""transformer (attention).

encoder: [Self-Attention, Feed-forward] x n
decoder: [Self-Attention, Source-Target-Attention, Feed-forward] x n
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention as comm_attn
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

from collections import namedtuple

import tensorflow as tf

EncoderState = namedtuple(
    'EncoderState',
    ['input', 'self_attn_bias', 'decoder_attn_bias', 'output'])
DecoderState = namedtuple(
    'DecoderState',
    ['input', 'self_attn_bias'])


def with_dropout(input, hparams):
    return tf.nn.dropout(input, 1.0 - hparams.layer_prepostprocess_dropout)


def get_ignore_padding(inputs):
    """
    Args:
        inputs: Tensor with shape [batch, memory_length, depth]
    """
    # Extract which individual embedding vectors are identically zero.
    # encoder_padding has shape [batch, memory_length].
    padding = comm_attn.embedding_to_padding(inputs)
    # ignore_padding has shape [batch, 1, 1, memory_length].
    # it also replaces all 1s in encoder_padding with -1e9 because idk.
    ignore_padding = comm_attn.attention_bias_ignore_padding(padding)
    return ignore_padding


@registry.register_model
class Transformer(t2t_model.T2TModel):
    """Attention net.  See file docstring.
    
    encoder: [Self-Attention, Feed-forward] x n
    decoder: [Self-Attention, Source-Target-Attention, Feed-forward] x n
    """

    def model_fn_body(self, features):
        """
        Args:
            features: 4D tensor.
        """
        hparams = self._hparams

        # Reshape (0, 1, 2, 3) => (0, 1 * 2, 3)
        # QUESTION: why?
        inputs = common_layers.flatten4d3d(features['inputs'])
        targets = common_layers.flatten4d3d(features['targets'])

        encoder = transformer_prepare_encoder(
            inputs, features['target_space_id'], hparams)
        encoder = transformer_encoder(encoder, hparams)

        decoder = transformer_prepare_decoder(targets, hparams)
        decoder_output = transformer_decoder(decoder, encoder, hparams)
        decoder_output = tf.expand_dims(decoder_output, 2)

        return decoder_output


@registry.register_model
class TransformerEncoder(t2t_model.T2TModel):
    """Transformer, encoder only."""

    def model_fn_body(self, features):
        hparams = self._hparams
        inputs = common_layers.flatten4d3d(features['inputs'])
        encoder = transformer_prepare_encoder(
            inputs, features['target_space_id'], hparams)
        encoder = transformer_encoder(encoder, hparams)
        return encoder.output


@registry.register_model
class TransformerDecoder(t2t_model.T2TModel):
    """Transformer, decoder only."""

    def model_fn_body(self, features):
        hparams = self._hparams
        targets = common_layers.flatten4d3d(features['targets'])
        decoder = transformer_prepare_decoder(targets, hparams)
        decoder_output = transformer_decoder(decoder, None, hparams)
        decoder_output = tf.expand_dims(decoder_output, 2)
        return decoder_output


def transformer_prepare_encoder(inputs, target_space, hparams):
    """Prepare one shard of the model for the encoder.
  
    Args:
      inputs: Tensor with shape [batch, memory_length, depth]
      target_space: a Tensor.
      hparams: run hyperparameters
  
    Returns:
      encoder_input: a Tensor, bottom of encoder stack
      encoder_self_attention_bias: a bias tensor for use in encoder self-attention
      encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
        attention
    """

    ignore_padding = get_ignore_padding(inputs)
    encoder_self_attention_bias = ignore_padding

    # Bias for self-attention to encourage attention to close positions.
    if hparams.proximity_bias:
        encoder_self_attention_bias += comm_attn.attention_bias_proximal(
            length=tf.shape(inputs)[1])

    # Append target_space_id embedding to inputs.
    emb_target_space = common_layers.embedding(
        x=target_space,
        vocab_size=32,
        dense_size=inputs.shape.as_list[-1],
        name='target_space_embedding')
    emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])

    # Question: wat
    encoder_input = inputs + emb_target_space
    if hparams.pos == 'timing':
        encoder_input = comm_attn.add_timing_signal_1d(encoder_input)
    # Putting this here since always called immediately after...
    encoder_input = with_dropout(encoder_input, hparams)

    return EncoderState(
        input=encoder_input,
        self_attn_bias=encoder_self_attention_bias,
        decoder_attn_bias=ignore_padding,
        output=None)


def transformer_prepare_decoder(targets, hparams):
    """Prepare one shard of the model for the decoder.
  
    Args:
      targets: a Tensor.
      hparams: run hyperparameters
  
    Returns:
      decoder_input: a Tensor, bottom of decoder stack
      decoder_self_attention_bias: a bias tensor for use in encoder self-attention
    """
    decoder_self_attention_bias = (
        comm_attn.attention_bias_lower_triangle(tf.shape(targets)[1]))
    if hparams.proximity_bias:
        decoder_self_attention_bias += comm_attn.attention_bias_proximal(
            tf.shape(targets)[1])
    decoder_input = common_layers.shift_left_3d(targets)
    if hparams.pos == 'timing':
        decoder_input = comm_attn.add_timing_signal_1d(decoder_input)
    # Putting this here since always called immediately after...
    decoder_input = with_dropout(decoder_input, hparams)

    return DecoderState(
        input=decoder_input,
        self_attn_bias=decoder_self_attention_bias)


def transformer_encoder(encoder_state,
                        hparams,
                        name='encoder'):
    """A stack of transformer layers.
  
    Args:
      encoder_input: a Tensor
      encoder_self_attention_bias: bias Tensor for self-attention
         (see comm_attn.attention_bias())
      hparams: hyperparameters for model
      name: a string
  
    Returns:
      y: a Tensors
    """
    x = encoder_state.input
    total_key_depth = hparams.attention_key_channels or hparams.hidden_size
    total_value_depth = hparams.attention_value_channels or hparams.hidden_size
    with tf.variable_scope(name):
        for layer in xrange(hparams.num_encoder_layers):
            with tf.variable_scope('layer_%d' % layer):

                # Multi-Head Attention Layer.
                with tf.variable_scope('self_attention'):
                    y = comm_attn.multihead_attention(
                        query_antecedent=common_layers.layer_preprocess(x, hparams),
                        memory_antecedent=None,
                        bias=encoder_state.self_attn_bias,
                        total_key_depth=total_key_depth,
                        total_value_depth=total_value_depth,
                        output_depth=hparams.hidden_size,
                        num_heads=hparams.num_heads,
                        dropout_rate=hparams.attention_dropout)
                    x = common_layers.layer_postprocess(x, y, hparams)

                # Feed-Forward Network Layer.
                with tf.variable_scope('ffn'):
                    y = transformer_ffn_layer(
                        common_layers.layer_preprocess(x, hparams), hparams)
                    x = common_layers.layer_postprocess(x, y, hparams)

    # if normalization is done in layer_preprocess, then it shuold also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    encoder_output = common_layers.layer_preprocess(x, hparams)
    return EncoderState(
        input=encoder_state.input,
        self_attn_bias=encoder_state.self_attn_bias,
        decoder_attn_bias=encoder_state.decoder_attn_bias,
        output=encoder_output)


def transformer_decoder(decoder_state,
                        encoder_state,
                        hparams,
                        name='decoder'):
    """A stack of transformer layers.
  
    Args:
      decoder_input: a Tensor
      encoder_output: a Tensor
      decoder_self_attention_bias: bias Tensor for self-attention
        (see comm_attn.attention_bias())
      encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
        (see comm_attn.attention_bias())
      hparams: hyperparameters for model
      name: a string
  
    Returns:
      y: a Tensors
    """
    x = decoder_state.input
    with tf.variable_scope(name):
        for layer in xrange(hparams.num_decoder_layers):
            with tf.variable_scope('layer_%d' % layer):
                with tf.variable_scope('self_attention'):
                    y = comm_attn.multihead_attention(
                        common_layers.layer_preprocess(
                            x, hparams), None, decoder_state.self_attn_bias,
                        hparams.attention_key_channels or hparams.hidden_size,
                        hparams.attention_value_channels or hparams.hidden_size,
                        hparams.hidden_size, hparams.num_heads,
                        hparams.attention_dropout)
                    x = common_layers.layer_postprocess(x, y, hparams)

                if encoder_state.output is not None:
                    with tf.variable_scope('encdec_attention'):
                        y = comm_attn.multihead_attention(
                            common_layers.layer_preprocess(
                                x, hparams), encoder_state.output,
                            encoder_state.decoder_attn_bias,
                            hparams.attention_key_channels or hparams.hidden_size,
                            hparams.attention_value_channels or hparams.hidden_size,
                            hparams.hidden_size, hparams.num_heads,
                            hparams.attention_dropout)
                        x = common_layers.layer_postprocess(x, y, hparams)
                with tf.variable_scope('ffn'):
                    y = transformer_ffn_layer(
                        common_layers.layer_preprocess(x, hparams), hparams)
                    x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it shuold also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return common_layers.layer_preprocess(x, hparams)


def transformer_ffn_layer(x, hparams):
    """Feed-forward layer in the transformer.
  
    Args:
      x: a Tensor of shape [batch_size, length, hparams.hidden_size]
      hparams: hyperparmeters for model
  
    Returns:
      a Tensor of shape [batch_size, length, hparams.hidden_size]
    """
    if hparams.ffn_layer == 'conv_hidden_relu':
        return common_layers.conv_hidden_relu(
            x,
            hparams.filter_size,
            hparams.hidden_size,
            dropout=hparams.relu_dropout)
    elif hparams.ffn_layer == 'parameter_attention':
        return comm_attn.parameter_attention(
            x, hparams.parameter_attention_key_channels or hparams.hidden_size,
               hparams.parameter_attention_value_channels or hparams.hidden_size,
            hparams.hidden_size, hparams.filter_size, hparams.num_heads,
            hparams.attention_dropout)
    elif hparams.ffn_layer == 'conv_hidden_relu_with_sepconv':
        return common_layers.conv_hidden_relu(
            x,
            hparams.filter_size,
            hparams.hidden_size,
            kernel_size=(3, 1),
            second_kernel_size=(31, 1),
            padding='LEFT',
            dropout=hparams.relu_dropout)
    else:
        assert hparams.ffn_layer == 'none'
        return x


@registry.register_hparams
def transformer_base():
    """Set of hyperparameters."""
    hparams = common_hparams.basic_params1()
    hparams.norm_type = 'layer'
    hparams.hidden_size = 512
    hparams.batch_size = 4096
    hparams.max_length = 256
    hparams.dropout = 0.0
    hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
    hparams.optimizer_adam_epsilon = 1e-9
    hparams.learning_rate_decay_scheme = 'noam'
    hparams.learning_rate = 0.1
    hparams.learning_rate_warmup_steps = 4000
    hparams.initializer_gain = 1.0
    hparams.num_hidden_layers = 6
    hparams.initializer = 'uniform_unit_scaling'
    hparams.weight_decay = 0.0
    hparams.optimizer_adam_beta1 = 0.9
    hparams.optimizer_adam_beta2 = 0.98
    hparams.num_sampled_classes = 0
    hparams.label_smoothing = 0.1
    hparams.shared_embedding_and_softmax_weights = int(True)

    hparams.add_hparam('filter_size', 2048)  # Add new ones like this.
    # layer-related flags
    hparams.add_hparam('num_encoder_layers', hparams.num_hidden_layers)
    hparams.add_hparam('num_decoder_layers', hparams.num_hidden_layers)
    # attention-related flags
    hparams.add_hparam('num_heads', 8)
    hparams.add_hparam('attention_key_channels', 0)
    hparams.add_hparam('attention_value_channels', 0)
    hparams.add_hparam('ffn_layer', 'conv_hidden_relu')
    hparams.add_hparam('parameter_attention_key_channels', 0)
    hparams.add_hparam('parameter_attention_value_channels', 0)
    # All hyperparameters ending in 'dropout' are automatically set to 0.0
    # when not in training mode.
    hparams.add_hparam('attention_dropout', 0.0)
    hparams.add_hparam('relu_dropout', 0.0)
    hparams.add_hparam('pos', 'timing')  # timing, none
    hparams.add_hparam('nbr_decoder_problems', 1)
    hparams.add_hparam('proximity_bias', int(False))
    return hparams


@registry.register_hparams
def transformer_n_da():
    """Normalize on layer input, instead of after residual connection.
  
    This version seems to cure failure-to-learn bugs - for example, with very
    deep networks or hard-to-learn mappings.
  
    Probably this should become the default.
  
    Returns:
      a hyperparameters.
    """
    hparams = transformer_base()
    hparams.layer_preprocess_sequence = 'n'
    hparams.layer_postprocess_sequence = 'da'
    # This version seems to benefit from a higher learning rate.
    hparams.learning_rate = 0.4
    return hparams


@registry.register_hparams
def transformer_n_da_l10():
    hparams = transformer_n_da()
    hparams.num_hidden_layers = 10
    return hparams


@registry.register_hparams
def transformer_big():
    """HParams for transfomer big model on WMT."""
    hparams = transformer_base()
    hparams.hidden_size = 1024
    hparams.filter_size = 4096
    hparams.num_heads = 16
    hparams.layer_prepostprocess_dropout = 0.3
    return hparams


@registry.register_hparams
def transformer_big_single_gpu():
    """HParams for transformer big model for single gpu."""
    hparams = transformer_big()
    hparams.layer_prepostprocess_dropout = 0.1
    hparams.learning_rate_warmup_steps = 16000
    hparams.optimizer_adam_beta2 = 0.998
    return hparams


@registry.register_hparams
def transformer_base_single_gpu():
    """HParams for transformer base model for single gpu."""
    hparams = transformer_base()
    hparams.batch_size = 2048
    hparams.learning_rate_warmup_steps = 16000
    return hparams


@registry.register_hparams
def transformer_parsing_base():
    """Hparams for parsing on wsj only."""
    hparams = transformer_base()
    hparams.attention_dropout = 0.2
    hparams.layer_prepostprocess_dropout = 0.2
    hparams.max_length = 512
    hparams.learning_rate_warmup_steps = 16000
    hparams.hidden_size = 1024
    hparams.learning_rate = 0.05
    hparams.shared_embedding_and_softmax_weights = int(False)
    return hparams


@registry.register_hparams
def transformer_parsing_big():
    """HParams for parsing on wsj semi-supervised."""
    hparams = transformer_big()
    hparams.max_length = 512
    hparams.shared_source_target_embedding = int(False)
    hparams.learning_rate_warmup_steps = 4000
    hparams.layer_prepostprocess_dropout = 0.1
    hparams.batch_size = 2048
    hparams.learning_rate = 0.05
    return hparams


@registry.register_hparams
def transformer_parsing_ice():
    """Hparams for parsing and tagging Icelandic text."""
    hparams = transformer_base_single_gpu()
    hparams.batch_size = 4096
    hparams.shared_embedding_and_softmax_weights = int(False)
    return hparams


@registry.register_hparams
def transformer_tiny():
    hparams = transformer_base()
    hparams.num_hidden_layers = 2
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 4
    return hparams


@registry.register_hparams
def transformer_small():
    hparams = transformer_base()
    hparams.num_hidden_layers = 2
    hparams.hidden_size = 256
    hparams.filter_size = 1024
    hparams.num_heads = 4
    return hparams


@registry.register_hparams
def transformer_l2():
    hparams = transformer_base()
    hparams.num_hidden_layers = 2
    return hparams


@registry.register_hparams
def transformer_l4():
    hparams = transformer_base()
    hparams.num_hidden_layers = 4
    return hparams


@registry.register_hparams
def transformer_l8():
    hparams = transformer_base()
    hparams.num_hidden_layers = 8
    return hparams


@registry.register_hparams
def transformer_l10():
    hparams = transformer_base()
    hparams.num_hidden_layers = 10
    return hparams


@registry.register_hparams
def transformer_h1():
    hparams = transformer_base()
    hparams.num_heads = 1
    return hparams


@registry.register_hparams
def transformer_h4():
    hparams = transformer_base()
    hparams.num_heads = 4
    return hparams


@registry.register_hparams
def transformer_h16():
    hparams = transformer_base()
    hparams.num_heads = 16
    return hparams


@registry.register_hparams
def transformer_h32():
    hparams = transformer_base()
    hparams.num_heads = 32
    return hparams


@registry.register_hparams
def transformer_k128():
    hparams = transformer_base()
    hparams.attention_key_channels = 128
    return hparams


@registry.register_hparams
def transformer_k256():
    hparams = transformer_base()
    hparams.attention_key_channels = 256
    return hparams


@registry.register_hparams
def transformer_ff1024():
    hparams = transformer_base()
    hparams.filter_size = 1024
    return hparams


@registry.register_hparams
def transformer_ff4096():
    hparams = transformer_base()
    hparams.filter_size = 4096
    return hparams


@registry.register_hparams
def transformer_dr0():
    hparams = transformer_base()
    hparams.layer_prepostprocess_dropout = 0.0
    return hparams


@registry.register_hparams
def transformer_dr2():
    hparams = transformer_base()
    hparams.layer_prepostprocess_dropout = 0.2
    return hparams


@registry.register_hparams
def transformer_ls0():
    hparams = transformer_base()
    hparams.label_smoothing = 0.0
    return hparams


@registry.register_hparams
def transformer_ls2():
    hparams = transformer_base()
    hparams.label_smoothing = 0.2
    return hparams


@registry.register_hparams
def transformer_hs256():
    hparams = transformer_base()
    hparams.hidden_size = 256
    return hparams


@registry.register_hparams
def transformer_hs1024():
    hparams = transformer_base()
    hparams.hidden_size = 1024
    return hparams


@registry.register_hparams
def transformer_big_dr1():
    hparams = transformer_base()
    hparams.hidden_size = 1024
    hparams.filter_size = 4096
    hparams.num_heads = 16
    hparams.layer_prepostprocess_dropout = 0.1
    return hparams


@registry.register_hparams
def transformer_big_enfr():
    hparams = transformer_big_dr1()
    hparams.shared_embedding_and_softmax_weights = int(False)
    hparams.filter_size = 8192
    hparams.layer_prepostprocess_dropout = 0.1
    return hparams


@registry.register_hparams
def transformer_big_dr2():
    hparams = transformer_big_dr1()
    hparams.layer_prepostprocess_dropout = 0.2
    return hparams


@registry.register_hparams
def transformer_parameter_attention_a():
    hparams = transformer_base()
    hparams.ffn_layer = 'parameter_attention'
    hparams.filter_size = 1536
    return hparams


@registry.register_hparams
def transformer_parameter_attention_b():
    hparams = transformer_base()
    hparams.ffn_layer = 'parameter_attention'
    hparams.filter_size = 512
    hparams.parameter_attention_key_channels = 1024
    hparams.parameter_attention_value_channels = 1024
    hparams.num_heads = 16
    return hparams


@registry.register_ranged_hparams('transformer_base')
def transformer_base_range(rhp):
    """Small range of hyperparameters."""
    hparams = transformer_base()
    common_hparams.fill_ranged_hparams_from_hparams(hparams, rhp)
    # After starting from base, set intervals for some parameters.
    rhp.set_float('learning_rate', 0.3, 3.0, scale=rhp.LOG_SCALE)
    rhp.set_discrete('learning_rate_warmup_steps',
                     [1000, 2000, 4000, 8000, 16000])
    rhp.set_float('initializer_gain', 0.5, 2.0)
    rhp.set_float('optimizer_adam_beta2', 0.85, 0.95)
    rhp.set_float('optimizer_adam_beta2', 0.97, 0.99)
    rhp.set_float('weight_decay', 0.0, 2.0)
