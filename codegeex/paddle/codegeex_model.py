import math
import paddle
import paddle.nn.functional as F


def fast_gelu(x):
    """Mindspore's fast gelu implementation."""
    return (
        x
        / (1 + paddle.exp(-1.702 * paddle.abs(x)))
        * paddle.exp(0.851 * (x - paddle.abs(x)))
    )


class MLP(paddle.nn.Layer):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(
        self,
        hidden_size,
    ):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        # Project to 4h.
        self.dense_h_to_4h = paddle.nn.Linear(
            self.hidden_size,
            4 * self.hidden_size,
        )

        self.activation_func = fast_gelu

        # Project back to h.
        self.dense_4h_to_h = paddle.nn.Linear(
            4 * self.hidden_size,
            self.hidden_size,
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)

        return output


class SelfAttention(paddle.nn.Layer):
    """self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        layer_number,
        fp16=True,
        attention_softmax_in_fp32=True,
    ):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.fp16 = fp16
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.layer_number = max(1, layer_number)

        assert self.hidden_size % self.num_attention_heads == 0
        self.hidden_size_per_attention_head = int(
            self.hidden_size // self.num_attention_heads
        )

        self.query = paddle.nn.Linear(self.hidden_size, self.hidden_size)
        self.key = paddle.nn.Linear(self.hidden_size, self.hidden_size)
        self.value = paddle.nn.Linear(self.hidden_size, self.hidden_size)

        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        self.softmax = paddle.nn.Softmax(axis=-1)

        self.dense = paddle.nn.Linear(self.hidden_size, self.hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_past=None,
        get_key_value=False,
        prompt_length=None,
        context_length=None,
    ):
        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        new_query_layer_shape = query_layer.shape[:-1] + [
            self.num_attention_heads,
            self.hidden_size_per_attention_head,
        ]
        query_layer = query_layer.reshape(new_query_layer_shape)

        new_query_layer_shape = key_layer.shape[:-1] + [
            self.num_attention_heads,
            self.hidden_size_per_attention_head,
        ]
        key_layer = key_layer.reshape(new_query_layer_shape)

        new_query_layer_shape = value_layer.shape[:-1] + [
            self.num_attention_heads,
            self.hidden_size_per_attention_head,
        ]
        value_layer = value_layer.reshape(new_query_layer_shape)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = paddle.concat(
                (past_key.cast(key_layer.dtype), key_layer), axis=0
            )
            value_layer = paddle.concat(
                (past_value.cast(value_layer.dtype), value_layer), axis=0
            )
        if get_key_value:
            present = (key_layer, value_layer)

        # ===================================
        # Raw attention scores. [b, np, sq, sk]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.shape[1],
            query_layer.shape[2],
            query_layer.shape[0],
            key_layer.shape[0],
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.reshape(
            [output_size[2], output_size[0] * output_size[1], -1]
        )
        key_layer = key_layer.reshape(
            [output_size[3], output_size[0] * output_size[1], -1]
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = (
            paddle.matmul(
                query_layer.transpose([1, 0, 2]),
                key_layer.transpose([1, 0, 2]).transpose([0, 2, 1]),
            )
            / self.norm_factor
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.reshape(output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if get_key_value:
            with paddle.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[
                        ..., attention_scores.shape[3] - 1, : attention_scores.shape[3]
                    ].unsqueeze(2)
                else:
                    attention_mask = attention_mask[
                        ..., : attention_scores.shape[3], : attention_scores.shape[3]
                    ]

        if context_length is not None:
            attention_mask = paddle.clone(attention_mask)
            attention_mask[:, :, context_length:, :] = True

        # attention scores and attention mask [b, np, sq, sk]
        # attention_scores = attention_mask_func(attention_scores, attention_mask)
        attention_scores = attention_scores - attention_mask * 10000.0
        if self.attention_softmax_in_fp32:
            attention_probs = self.softmax(attention_scores.cast("float32")).cast(
                "float16"
            )
        else:
            attention_probs = self.softmax(attention_scores)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sq, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.shape[1],
            value_layer.shape[2],
            query_layer.shape[0],
            value_layer.shape[3],
        )

        # change view [sq, b * np, hn]
        value_layer = value_layer.reshape(
            [value_layer.shape[0], output_size[0] * output_size[1], -1]
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.reshape(
            [output_size[0] * output_size[1], output_size[2], -1]
        )

        context_layer = paddle.bmm(
            attention_probs, value_layer.unsqueeze(0).transpose([0, 2, 1, 3]).squeeze(0)
        )

        # change view [b, np, sq, hn]
        context_layer = context_layer.reshape(output_size)

        # # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.transpose([2, 0, 1, 3])

        # # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.shape[:-2] + [
            self.hidden_size,
        ]
        context_layer = context_layer.reshape(new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output = self.dense(context_layer)

        if get_key_value:
            output = [output, present]

        return output


class TopQuerySelfAttention(paddle.nn.Layer):
    """Top query self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        layer_number,
        fp16=True,
        attention_softmax_in_fp32=True,
    ):
        super(TopQuerySelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.fp16 = fp16
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.layer_number = max(1, layer_number)

        assert self.hidden_size % self.num_attention_heads == 0
        self.hidden_size_per_attention_head = int(
            self.hidden_size // self.num_attention_heads
        )

        self.query = paddle.nn.Linear(self.hidden_size, self.hidden_size)
        self.key = paddle.nn.Linear(self.hidden_size, self.hidden_size)
        self.value = paddle.nn.Linear(self.hidden_size, self.hidden_size)

        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        self.softmax = paddle.nn.Softmax(axis=-1)

        self.dense = paddle.nn.Linear(self.hidden_size, self.hidden_size)

    def forward(
        self,
        hidden_states,
        query_hidden_state,
        attention_mask,
        layer_past=None,
        get_key_value=False,
        prompt_length=None,
        context_length=None,
    ):

        # hidden_states: [sq, b, h]
        query_layer = self.query(query_hidden_state)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        new_query_layer_shape = query_layer.shape[:-1] + [
            self.num_attention_heads,
            self.hidden_size_per_attention_head,
        ]
        query_layer = query_layer.reshape(new_query_layer_shape)

        new_query_layer_shape = key_layer.shape[:-1] + [
            self.num_attention_heads,
            self.hidden_size_per_attention_head,
        ]
        key_layer = key_layer.reshape(new_query_layer_shape)

        new_query_layer_shape = value_layer.shape[:-1] + [
            self.num_attention_heads,
            self.hidden_size_per_attention_head,
        ]
        value_layer = value_layer.reshape(new_query_layer_shape)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = paddle.concat(
                (past_key.cast(key_layer.dtype), key_layer), axis=0
            )
            value_layer = paddle.concat(
                (past_value.cast(value_layer.dtype), value_layer), axis=0
            )
        if get_key_value:
            present = (key_layer, value_layer)

        # ===================================
        # Raw attention scores. [b, np, sq, sk]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.shape[1],
            query_layer.shape[2],
            query_layer.shape[0],
            key_layer.shape[0],
        )

        # [s, b, np, hn] -> [s, b * np, hn]
        query_layer = query_layer.reshape(
            [output_size[2], output_size[0] * output_size[1], -1]
        )
        key_layer = key_layer.reshape(
            [output_size[3], output_size[0] * output_size[1], -1]
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = (
            paddle.matmul(
                query_layer.transpose([1, 0, 2]),
                key_layer.transpose([1, 0, 2]).transpose([0, 2, 1]),
            )
            / self.norm_factor
        )

        # change view to [b, np, s, s]
        attention_scores = matmul_result.reshape(output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if get_key_value:
            with paddle.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[
                        ..., attention_scores.shape[3] - 1, : attention_scores.shape[3]
                    ].unsqueeze(2)
                else:
                    attention_mask = attention_mask[
                        ..., : attention_scores.shape[3], : attention_scores.shape[3]
                    ]

        if context_length is not None:
            attention_mask = paddle.clone(attention_mask)
            attention_mask[:, :, context_length:, :] = True

        # attention scores and attention mask [b, np, sq, sk]
        # attention_scores = attention_mask_func(attention_scores, attention_mask)
        attention_scores = attention_scores - attention_mask * 10000.0
        if self.attention_softmax_in_fp32:
            attention_probs = self.softmax(attention_scores.cast("float32")).cast(
                "float16"
            )
        else:
            attention_probs = self.softmax(attention_scores)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sq, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.shape[1],
            value_layer.shape[2],
            query_layer.shape[0],
            value_layer.shape[3],
        )

        # change view [sq, b * np, hn]
        value_layer = value_layer.reshape(
            [value_layer.shape[0], output_size[0] * output_size[1], -1]
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.reshape(
            [output_size[0] * output_size[1], output_size[2], -1]
        )

        # matmul: [b * np, sq, hn]
        context_layer = paddle.bmm(
            attention_probs, value_layer.unsqueeze(0).transpose([0, 2, 1, 3]).squeeze(0)
        )

        # change view [b, np, sq, hn]
        context_layer = context_layer.reshape(output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.transpose([2, 0, 1, 3])

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.shape[:-2] + [
            self.hidden_size,
        ]
        context_layer = context_layer.reshape(new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output = self.dense(context_layer)

        if get_key_value:
            output = [output, present]

        return output


class TransformerLayer(paddle.nn.Layer):
    """A single transformer layer.

    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        layer_number,
        layernorm_epsilon=1e-5,
        fp16=True,
        attention_softmax_in_fp32=True,
    ):
        super(TransformerLayer, self).__init__()
        self.hidden_size = hidden_size
        self.layernorm_epsilon = layernorm_epsilon
        self.layer_number = layer_number

        # Layernorm on the input data.
        self.input_layernorm = paddle.nn.LayerNorm(
            hidden_size, epsilon=self.layernorm_epsilon
        )

        # Self attention.
        self.attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            layer_number,
            fp16,
            attention_softmax_in_fp32,
        )

        # Layernorm on the input data.
        self.post_attention_layernorm = paddle.nn.LayerNorm(
            self.hidden_size, epsilon=self.layernorm_epsilon
        )
        self.mlp = MLP(self.hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_past=None,
        get_key_value=False,
        prompt_length=None,
        context_length=None,
    ):
        # hidden_states: [b, s, h]
        # Use FP32 for Layernorm
        # layernorm_output = self.input_layernorm(hidden_states.cast("float32")).cast("float16")
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output = self.attention(
            layernorm_output,
            attention_mask,
            layer_past=layer_past,
            get_key_value=get_key_value,
            prompt_length=prompt_length,
            context_length=context_length,
        )

        if get_key_value:
            attention_output, presents = attention_output

        # Residual connection.
        residual = hidden_states
        layernorm_input = attention_output + residual

        # Use FP32 for Layernorm
        # layernorm_output = self.post_attention_layernorm(layernorm_input.cast("float32")).cast("float16")
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        mlp_output = self.mlp(layernorm_output)
        output = mlp_output + layernorm_input

        if get_key_value:
            output = [output, presents]

        return output


class TopQueryLayer(paddle.nn.Layer):
    """A single top query layer.

    Top query layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        layer_number,
        layernorm_epsilon=1e-5,
    ):
        super(TopQueryLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.layernorm_epsilon = layernorm_epsilon
        self.layer_number = layer_number

        # Use FP32 for Layernorm
        self.input_layernorm = paddle.nn.LayerNorm(
            self.hidden_size, epsilon=self.layernorm_epsilon
        )

        # Self attention.
        self.attention = TopQuerySelfAttention(
            self.hidden_size, self.num_attention_heads, self.layer_number
        )
        # Layernorm on the input data.
        self.post_attention_layernorm = paddle.nn.LayerNorm(
            self.hidden_size, epsilon=self.layernorm_epsilon
        )

        # MLP
        self.mlp = MLP(self.hidden_size)

    def forward(
        self,
        hidden_states,
        query_hidden_state,
        attention_mask,
        layer_past=None,
        get_key_value=False,
        prompt_length=None,
        context_length=None,
    ):
        # hidden_states: [b, s, h]
        # assert query_hidden_state != None

        # Use FP32 for Layernorm
        # layernorm_output = self.input_layernorm(hidden_states.cast("float32")).cast("float16")
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output = self.attention(
            layernorm_output,
            query_hidden_state,
            attention_mask,
            layer_past=layer_past,
            get_key_value=get_key_value,
            prompt_length=prompt_length,
            context_length=context_length,
        )

        if get_key_value:
            attention_output, presents = attention_output

        # Residual connection.
        residual = hidden_states
        layernorm_input = attention_output + residual

        # Use FP32 for Layernorm
        # layernorm_output = self.post_attention_layernorm(layernorm_input.cast("float32")).cast("float16")
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        residual = layernorm_input
        output = mlp_output + residual

        if get_key_value:
            output = [output, presents]

        return output


class Transformer(paddle.nn.Layer):
    """Transformer class."""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_layers,
        layernorm_epsilon=1e-5,
    ):
        super(Transformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.layernorm_epsilon = layernorm_epsilon
        # Number of layers:
        self.num_layers = num_layers
        self.num_unique_layers = None

        #################
        assert self.num_unique_layers is None
        #################

        if self.num_unique_layers is None:
            self.num_unique_layers = self.num_layers
        assert (
            self.num_layers % self.num_unique_layers == 0
        ), "number of layers should be divisible by number of unique layers"

        # Transformer layers.
        def build_layer(layer_number):
            return TransformerLayer(
                self.hidden_size, self.num_attention_heads, layer_number
            )

        self.layers = paddle.nn.LayerList(
            [build_layer(i + 1) for i in range(self.num_unique_layers)]
        )

        self.topQueryLayer = TopQueryLayer(
            self.hidden_size, self.num_attention_heads, self.num_unique_layers
        )

        self.final_layernorm = paddle.nn.LayerNorm(
            self.hidden_size, epsilon=self.layernorm_epsilon
        )

    def _get_layer_index(self, layer_number):
        return layer_number % self.num_unique_layers

    def _get_layer(self, layer_number):
        return self.layers[self._get_layer_index(layer_number)]

    def forward(
        self,
        hidden_states,
        query_hidden_state,
        attention_mask,
        layer_past=None,
        get_key_value=False,
        prompt_length=None,
        context_length=None,
    ):
        # data format change to avoid explicit tranposes : [b s h] --> [s b h]
        hidden_states = hidden_states.transpose([1, 0, 2])
        query_hidden_state = query_hidden_state.transpose([1, 0, 2])

        if get_key_value:
            presents = []
        for index in range(self.num_layers):
            layer = self._get_layer(index)
            past = None
            if layer_past is not None:
                past = layer_past[index]
            hidden_states = layer(
                hidden_states,
                attention_mask,
                layer_past=past,
                get_key_value=get_key_value,
                prompt_length=prompt_length,
                context_length=context_length,
            )
            if get_key_value:
                hidden_states, present = hidden_states
                presents.append(present)

        # Use FP32 for Layernorm
        # hidden_states_ = self.final_layernorm(hidden_states.cast("float32")).cast("float16")
        hidden_states_ = self.final_layernorm(hidden_states)

        #################################
        # top query layer
        #################################
        past = None
        if layer_past is not None:
            past = layer_past[self.num_layers]
        hidden_states = self.topQueryLayer(
            hidden_states_,
            query_hidden_state,
            attention_mask,
            layer_past=past,
            get_key_value=get_key_value,
            prompt_length=prompt_length,
            context_length=context_length,
        )

        if get_key_value:
            hidden_states, present = hidden_states
            presents.append(present)

        # reverting data format change [s b h] --> [b s h]
        output = hidden_states.transpose([1, 0, 2])

        if get_key_value:
            output = [output, presents]

        return output

    def state_dict_for_save_checkpoint(
        self, destination=None, prefix="", keep_vars=False
    ):
        return self.state_dict(destination, prefix, keep_vars)


class Embedding(paddle.nn.Layer):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
    """

    def __init__(
        self,
        hidden_size,
        vocab_size,
        max_sequence_length,
    ):
        super(Embedding, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length

        # Word embeddings.
        self.word_embeddings = paddle.nn.Embedding(self.vocab_size, self.hidden_size)
        self._word_embeddings_key = "word_embeddings"

        # Position embedding.
        self.position_embeddings = paddle.nn.Embedding(
            self.max_sequence_length, self.hidden_size
        )
        self.position_embeddings = self.position_embeddings.to(dtype="float16")
        self._position_embeddings_key = "position_embeddings"

    def forward(self, input_ids, position_ids):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings

        return embeddings

    def state_dict_for_save_checkpoint(
        self, destination=None, prefix="", keep_vars=False
    ):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._word_embeddings_key] = self.word_embeddings.state_dict(
            destination, prefix, keep_vars
        )
        state_dict_[
            self._position_embeddings_key
        ] = self.position_embeddings.state_dict(destination, prefix, keep_vars)

        return state_dict_

    def set_state_dict(self, state_dict, use_structured_name=True):
        """Customized load."""

        # Word embedding.
        if self._word_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if "word_embeddings" in key:
                    state_dict_[key.split("word_embeddings.")[1]] = state_dict[key]
        state_dict_["weight"] = state_dict_["weight"][: self.vocab_size]
        self.word_embeddings.set_state_dict(
            state_dict_, use_structured_name=use_structured_name
        )

        # Position embedding.
        if self._position_embeddings_key in state_dict:
            state_dict_ = state_dict[self._position_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if "position_embeddings" in key:
                    state_dict_[key.split("position_embeddings.")[1]] = state_dict[key]
        self.position_embeddings.set_state_dict(
            state_dict_, use_structured_name=use_structured_name
        )


class QueryEmbedding(paddle.nn.Layer):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
    """

    def __init__(
        self,
        hidden_size,
        vocab_size,
        max_sequence_length,
    ):
        super(QueryEmbedding, self).__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length

        # Top query position embedding (serial).
        self.top_query_embeddings = paddle.nn.Embedding(
            self.max_sequence_length, self.hidden_size
        )
        self.top_query_embeddings = self.top_query_embeddings.to(dtype="float16")
        self._top_query_embeddings_key = "top_query_embeddings"

    def forward(self, position_ids):
        # Embeddings.
        embeddings = self.top_query_embeddings(position_ids)

        return embeddings

    def state_dict_for_save_checkpoint(
        self, destination=None, prefix="", keep_vars=False
    ):
        """For easy load."""

        state_dict_ = {}
        state_dict_[
            self._top_query_embeddings_key
        ] = self.top_query_embeddings.state_dict(destination, prefix, keep_vars)

        return state_dict_

    def set_state_dict(self, state_dict, use_structured_name=True):
        """Customized load."""

        # Position embedding.
        if self._top_query_embeddings_key in state_dict:
            state_dict_ = state_dict[self._top_query_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if "top_query_embeddings" in key:
                    state_dict_[key.split("top_query_embeddings.")[1]] = state_dict[key]
        self.top_query_embeddings.set_state_dict(
            state_dict_, use_structured_name=use_structured_name
        )


class TransformerLanguageModel(paddle.nn.Layer):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        attention_mask_func: a function that takes `unmaksed-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
          masked-attention-scores = attention_mask_func(
                                     unmaksed-attention-scores, attention-mask)
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
    """

    def __init__(
        self,
        hidden_size,
        num_layers,
        num_attention_heads,
        padded_vocab_size,
        max_position_embeddings,
    ):
        super(TransformerLanguageModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.padded_vocab_size = padded_vocab_size
        self.max_position_embeddings = max_position_embeddings

        # Embeddings
        self.embedding = Embedding(
            self.hidden_size, self.padded_vocab_size, self.max_position_embeddings
        )
        self._embedding_key = "embedding"

        # Query embeddings
        self.topQueryEmbedding = QueryEmbedding(
            self.hidden_size, self.padded_vocab_size, self.max_position_embeddings
        )
        self._topQueryEmbedding_key = "topQueryEmbedding"

        # Transformer
        self.transformer = Transformer(
            self.hidden_size, self.num_attention_heads, self.num_layers
        )
        self._transformer_key = "transformer"

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        layer_past=None,
        get_key_value=False,
        prompt_length=None,
        context_length=None,
    ):

        # Embeddings.
        embedding_output = self.embedding(input_ids, position_ids)
        query_position_ids = position_ids
        queryEmbedding_out = self.topQueryEmbedding(query_position_ids)

        # Transformer.
        transformer_output = self.transformer(
            embedding_output,
            queryEmbedding_out,
            attention_mask,
            layer_past=layer_past,
            get_key_value=get_key_value,
            prompt_length=prompt_length,
            context_length=context_length,
        )

        return transformer_output

    def state_dict_for_save_checkpoint(
        self, destination=None, prefix="", keep_vars=False
    ):
        """For easy load."""

        state_dict_ = {}
        state_dict_[
            self._embedding_key
        ] = self.embedding.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        state_dict_[
            self._topQueryEmbedding_key
        ] = self.topQueryEmbedding.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        state_dict_[
            self._transformer_key
        ] = self.transformer.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )

        return state_dict_

    def set_state_dict(self, state_dict, use_structured_name=True):
        """Customized load."""

        # Embedding.
        if self._embedding_key in state_dict:
            state_dict_ = state_dict[self._embedding_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if "_embeddings" in key:
                    state_dict_[key] = state_dict[key]
        self.embedding.set_state_dict(
            state_dict_, use_structured_name=use_structured_name
        )

        if self._topQueryEmbedding_key in state_dict:
            state_dict_ = state_dict[self._topQueryEmbedding_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if "_embeddings" in key:
                    state_dict_[key] = state_dict[key]
        self.topQueryEmbedding.set_state_dict(
            state_dict_, use_structured_name=use_structured_name
        )

        # Transformer.
        if self._transformer_key in state_dict:
            state_dict_ = state_dict[self._transformer_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if "transformer." in key:
                    state_dict_[key.split("transformer.")[1]] = state_dict[key]
        self.transformer.set_state_dict(
            state_dict_, use_structured_name=use_structured_name
        )


class CodeGeeXModel(paddle.nn.Layer):
    """CodeGeeX: A Multilingual Code Generation Model."""

    def __init__(
        self,
        hidden_size,
        num_layers,
        num_attention_heads,
        padded_vocab_size,
        max_position_embeddings,
    ):
        super(CodeGeeXModel, self).__init__()

        self.language_model = TransformerLanguageModel(
            hidden_size,
            num_layers,
            num_attention_heads,
            padded_vocab_size,
            max_position_embeddings,
        )
        self._language_model_key = "language_model"

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        layer_past=None,
        get_key_value=False,
        prompt_length=None,
        context_length=None,
    ):
        # Language model.
        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            layer_past=layer_past,
            get_key_value=get_key_value,
            prompt_length=prompt_length,
            context_length=context_length,
        )

        if get_key_value:
            lm_output, presents = lm_output

        output = F.linear(
            lm_output,
            self.language_model.embedding.word_embeddings.weight.cast(
                "float16"
            ).transpose([1, 0]),
        )

        if get_key_value:
            output = [output, presents]

        return output

    def state_dict_for_save_checkpoint(
        self, destination=None, prefix="", keep_vars=False
    ):

        state_dict_ = {}
        state_dict_[
            self._language_model_key
        ] = self.language_model.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )
        return state_dict_

    def set_state_dict(self, state_dict, use_structured_name=True):
        """Customized load."""

        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.set_state_dict(
            state_dict, use_structured_name=use_structured_name
        )
