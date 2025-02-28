import tensorflow as tf
import tensorflow_addons as tfa
class ViTEmbeddings(tf.keras.layers.Layer):
    def __init__(self, patch_size, hidden_size, dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.hidden_size = hidden_size

        """二维"""
        self.patch_embeddings = tf.keras.layers.Conv2D(
            filters=hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            dtype=tf.float32  # 混合精度确保权重为 float32
        )
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.hidden_size), trainable=True, name="cls_token", dtype=tf.float32
        )
        num_patches = (input_shape[1] // self.patch_size[0]) * (input_shape[2] // self.patch_size[1])  # [16,2]
        self.position_embeddings = self.add_weight(
            shape=(1, num_patches + 1, self.hidden_size), trainable=True, name="position_embeddings", dtype=tf.float32
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        inputs_shape = tf.shape(inputs)  # N,H,W,C
        embeddings = self.patch_embeddings(inputs, training=training)

        # Flatten the patches and adjust shape
        embeddings = tf.reshape(embeddings, [inputs_shape[0], -1, self.hidden_size])

        # 将 cls_token 和 position_embeddings 转换为与 inputs 相同的类型
        cls_tokens = tf.cast(self.cls_token, dtype=embeddings.dtype)
        position_embeddings = tf.cast(self.position_embeddings, dtype=embeddings.dtype)

        # print(f"Input dtype: {inputs.dtype}, CLS token dtype: {cls_tokens.dtype}, Positional embedding dtype: {position_embeddings.dtype}")
        # add the [CLS] token to the embedded patch tokens
        # 混合精度
        # cls_tokens = tf.repeat(self.cls_token, repeats=inputs_shape[0], axis=0)
        cls_tokens = tf.repeat(cls_tokens, repeats=inputs_shape[0], axis=0)
        embeddings = tf.concat((cls_tokens, embeddings), axis=1)

        # add positional encoding to each token
        # 混合精度
        # embeddings = embeddings + self.position_embeddings
        embeddings = embeddings + position_embeddings
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
        self.dense1 = tf.keras.layers.Dense(self.mlp_dim, dtype=tf.float32)
        self.activation1 = tf.keras.layers.Activation(self.activation)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense2 = tf.keras.layers.Dense(input_shape[-1] if self.out_dim is None else self.out_dim, dtype=tf.float32)

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
            dropout=0.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_before = tf.keras.layers.LayerNormalization(dtype=tf.float32)
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads,
            attention_dim // num_heads,
            use_bias=attention_bias,
            dropout=attention_dropout,
            dtype=tf.float32
        )
        self.stochastic_depth = tfa.layers.StochasticDepth(sd_survival_probability)
        self.norm_after = tf.keras.layers.LayerNormalization(dtype=tf.float32)
        self.mlp = MLP(mlp_dim=mlp_dim, activation=activation, dropout=dropout)

    def build(self, input_shape):
        super().build(input_shape)
        # TODO YONIGO: tf doc says to do this  ¯\_(ツ)_/¯
        self.attn._build_from_signature(input_shape, input_shape)

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
        self.norm = tf.keras.layers.LayerNormalization(dtype=tf.float32)
        self.head = tf.keras.layers.Dense(num_classes, dtype=tf.float32)

        sd = tf.linspace(1.0, sd_survival_probability, depth)

        self.blocks = [
            Block(
                num_heads,
                attention_dim=hidden_size,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                mlp_dim=mlp_dim,
                sd_survival_probability=(sd[i].numpy().item()),
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
        # x = self.embeddings(inputs, training=training)
        # for block in self.blocks:
        #     x = block(x, training=training)
        # x = self.norm(x)
        # x = x[:, 0]  # take only cls_token
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
        vitblocks_weights_softmax = tf.reshape(vitblocks_weights_softmax, [-1, 1, 1])
        weighted_output = tf.reduce_sum(vitblocks_weights_softmax * outputs_tensor, axis=0)

        return weighted_output

    def get_last_selfattention(self, inputs: tf.Tensor):
        x = self.embeddings(inputs, training=False)
        for block in self.blocks[:-1]:
            x = block(x, training=False)
        return self.blocks[-1].get_attention_scores(x)