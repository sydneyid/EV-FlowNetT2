import tensorflow as tf

_BASE_CHANNELS = 64

class BuildResNetBlock(tf.keras.layers.Layer):
    def __init__(self, channels_out, do_batch_norm=False, data_format='channels_last'):
        super().__init__()
        self.do_batch_norm = do_batch_norm
        self.data_format = data_format
        
        self.conv1 = tf.keras.layers.Conv2D(channels_out, 3, padding='same', data_format=data_format)
        self.conv2 = tf.keras.layers.Conv2D(channels_out, 3, padding='same', data_format=data_format)
        if do_batch_norm:
            axis = 1 if data_format == 'channels_first' else -1
            self.bn1 = tf.keras.layers.BatchNormalization(axis=axis)
            self.bn2 = tf.keras.layers.BatchNormalization(axis=axis)
        else:
            self.bn1 = None
            self.bn2 = None

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        if self.bn1:
            x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        if self.bn2:
            x = self.bn2(x, training=training)
        return tf.nn.relu(x + inputs)  # residual connection

class EVFlowNetModel(tf.keras.Model):
    def __init__(self, do_batch_norm=False, data_format='channels_last'):
        super().__init__()
        self.do_batch_norm = do_batch_norm
        self.data_format = data_format
        
        # Encoder: strided convolutions for downsampling
        self.encoder_convs = []
        self.encoder_bns = []
        for i in range(4):
            out_channels = int((2 ** i) * _BASE_CHANNELS)
            self.encoder_convs.append(
                tf.keras.layers.Conv2D(out_channels, 3, strides=2, padding='same', data_format=self.data_format)
            )
            if self.do_batch_norm:
                axis = 1 if data_format == 'channels_first' else -1
                self.encoder_bns.append(
                    tf.keras.layers.BatchNormalization(axis=axis)
                )
            else:
                self.encoder_bns.append(None)
        
        # Transition ResNet blocks
        self.transition_blocks = []
        for i in range(2):
            self.transition_blocks.append(
                BuildResNetBlock(8 * _BASE_CHANNELS, do_batch_norm=self.do_batch_norm, data_format=self.data_format)
            )
        
        # Decoder: upsampling with Conv2DTranspose
        self.decoder_deconvs = []
        self.decoder_bns = []
        self.flow_predictors = []
        for i in range(4):
            out_channels = int((2 ** (2 - i)) * _BASE_CHANNELS)
            self.decoder_deconvs.append(
                tf.keras.layers.Conv2DTranspose(out_channels, 3, strides=2, padding='same', data_format=self.data_format)
            )
            if self.do_batch_norm:
                axis = 1 if data_format == 'channels_first' else -1
                self.decoder_bns.append(
                    tf.keras.layers.BatchNormalization(axis=axis)
                )
            else:
                self.decoder_bns.append(None)
            self.flow_predictors.append(
                tf.keras.layers.Conv2D(2, 3, padding='same', data_format=self.data_format)
            )

    def call(self, inputs, training=False):
        x = inputs
        if self.data_format == 'channels_first':
            x = tf.transpose(x, [0, 3, 1, 2])
        
        skip_connections = []
        # Encoder: downsample at each layer
        for conv, bn in zip(self.encoder_convs, self.encoder_bns):
            x = conv(x)
            if bn:
                x = bn(x, training=training)
            x = tf.nn.relu(x)
            skip_connections.append(x)
        
        # Transition
        for block in self.transition_blocks:
            x = block(x, training=training)
        
        # Decoder: upsample and use skip connections
        flow_dict = {}
        for i in range(4):
            skip = skip_connections[3 - i]
            # Upsample x to match skip size if needed (should be same due to strides)
            x = tf.concat([x, skip], axis=-1)
            x = self.decoder_deconvs[i](x)
            if self.decoder_bns[i]:
                x = self.decoder_bns[i](x, training=training)
            x = tf.nn.relu(x)
            
            flow = self.flow_predictors[i](x) * 256.0
            if self.data_format == 'channels_first':
                flow = tf.transpose(flow, [0, 2, 3, 1])
            flow_dict[f'flow{i}'] = flow
            
            # Concatenate flow back to features before next iteration
            if self.data_format == 'channels_first':
                x = tf.concat([x, tf.transpose(flow, [0, 3, 1, 2])], axis=1)
            else:
                x = tf.concat([x, flow], axis=-1)
        
        return flow_dict