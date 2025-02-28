import tensorflow as tf
import tensorflow_addons as tfa

class LoRALayer(tf.keras.layers.Layer):
    def __init__(self, hidden_size, rank=8, alpha=16, **kwargs):
        super().__init__(**kwargs)
        self.scale = alpha / rank
        self.lora_A = tf.keras.layers.Dense(rank, use_bias=False)
        self.lora_B = tf.keras.layers.Dense(hidden_size, use_bias=False)

    def call(self, inputs):
        output = self.lora_B(self.lora_A(inputs))
        return tf.cast(self.scale, output.dtype) * output



class ViTEmbeddings(tf.keras.layers.Layer):
    def __init__(self, patch_size, hidden_size, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.patch_embeddings = tf.keras.layers.Conv2D(
            filters=hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
        )
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.hidden_size), trainable=True, name="cls_token"
        )
        num_patches = (input_shape[1] // self.patch_size[0]) * (input_shape[2] // self.patch_size[1])
        self.position_embeddings = self.add_weight(
            shape=(1, num_patches + 1, self.hidden_size), trainable=True, name="position_embeddings"
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        inputs_shape = tf.shape(inputs)
        embeddings = self.patch_embeddings(inputs, training=training)
        embeddings = tf.reshape(embeddings, [inputs_shape[0], -1, self.hidden_size])

        cls_tokens = tf.repeat(self.cls_token, repeats=inputs_shape[0], axis=0)
        embeddings = tf.concat((cls_tokens, embeddings), axis=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings, training=training)
        return embeddings



class MLP(tf.keras.layers.Layer):
    def __init__(self, mlp_dim, out_dim=None, activation="gelu", dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.activation = activation
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(self.mlp_dim)
        self.activation1 = tf.keras.layers.Activation(self.activation)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense2 = tf.keras.layers.Dense(input_shape[-1] if self.out_dim is None else self.out_dim)

    def call(self, inputs: tf.Tensor, training: bool = False):
        x = self.dense1(inputs)
        x = self.activation1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        return x


class Block(tf.keras.layers.Layer):
    def __init__(
            self,
            num_heads,
            attention_dim,
            attention_bias,
            mlp_dim,
            attention_dropout=0.0,
            sd_survival_probability=1.0,
            activation="gelu",
            rank=8,
            dropout=0.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_before = tf.keras.layers.LayerNormalization()
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads,
            attention_dim // num_heads,
            use_bias=attention_bias,
            dropout=attention_dropout,
        )
        self.lora = LoRALayer(attention_dim, rank=rank)
        self.stochastic_depth = tfa.layers.StochasticDepth(sd_survival_probability)
        self.norm_after = tf.keras.layers.LayerNormalization()
        self.mlp = MLP(mlp_dim=mlp_dim, activation=activation, dropout=dropout)

    def build(self, input_shape):
        super().build(input_shape)
        # TODO YONIGO: tf doc says to do this  ¯\_(ツ)_/¯
        self.attn._build_from_signature(input_shape, input_shape)

    #def call(self, inputs, training=False):
    #    x = self.norm_before(inputs, training=training)
    #    attention_output = self.attn(x, x, training=training)
    #    attention_output += self.lora(x)  # Apply LoRA here
    #    x = x + attention_output
    #    x = self.stochastic_depth([inputs, x], training=training)
    #    x2 = self.norm_after(x, training=training)
    #    x2 = self.mlp(x2, training=training)
    #    return self.stochastic_depth([x, x2], training=training)
    
    def call(self, inputs, training=False):
        x = self.norm_before(inputs, training=training)
        x = self.attn(x, x, training=training)
        x = self.stochastic_depth([inputs, x], training=training)
        x2 = self.norm_after(x, training=training)
        x2 = self.mlp(x2, training=training)
        return self.stochastic_depth([x, x2], training=training)

    def get_attention_scores(self, inputs):
        x = self.norm_before(inputs, training=False)
        _, weights = self.attn(x, x, training=False, return_attention_scores=True)
        return weights


class VisionTransformer(tf.keras.Model):
    def __init__(
            self,
            patch_size,
            hidden_size,
            depth,
            num_heads,
            mlp_dim,
            num_classes,
            rank,
            dropout=0.0,
            sd_survival_probability=1.0,
            attention_bias=False,
            attention_dropout=0.0,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.embeddings = [ViTEmbeddings(patch_size[i], hidden_size, dropout) for i in range(len(patch_size))]
        self.norm = tf.keras.layers.LayerNormalization()
        self.head = tf.keras.layers.Dense(num_classes)


        sd = tf.linspace(1.0, sd_survival_probability, depth)

        self.blocks = [
            Block(
                num_heads,
                attention_dim=hidden_size,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                mlp_dim=mlp_dim,
                sd_survival_probability=(sd[i].numpy().item()),
                rank=rank,
                dropout=dropout,
            )
            for i in range(depth)
        ]

        self.vitblocks = []
        for i in range(len(patch_size)):
            self.vitblocks.append([
                self.embeddings[i],
                self.blocks,
                self.norm,
                self.head
            ])

        self.vitblocks_weights = tf.Variable(tf.random.normal([len(patch_size)]), trainable=True)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        outputs = []
        for i, block in enumerate(self.vitblocks):
            x = block[0](inputs, training=training)
            for layer in block[1]:
                x = layer(x, training=training)
            x = block[2](x)
            x = x[:, 0]
            x = block[3](x)
            outputs.append(x)

        outputs_tensor = tf.stack(outputs, axis=0)
        vitblocks_weights_softmax = tf.nn.softmax(self.vitblocks_weights)

        # ✅ Explicitly cast vitblocks_weights_softmax to match dtype
        vitblocks_weights_softmax = tf.cast(tf.reshape(vitblocks_weights_softmax, [-1, 1, 1]), outputs_tensor.dtype)

        weighted_output = tf.reduce_sum(vitblocks_weights_softmax * outputs_tensor, axis=0)

        return weighted_output


    def get_last_selfattention(self, inputs: tf.Tensor):
        x = self.embeddings(inputs, training=False)
        for block in self.blocks[:-1]:
            x = block(x, training=False)
        return self.blocks[-1].get_attention_scores(x)
