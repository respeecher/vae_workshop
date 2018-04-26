import tensorflow as tf
import tflearn


def retrieve_seq_length_op(data):
    """ An op to compute the length of a sequence. 0 are masked. """
    with tf.name_scope('GetLength'):
        used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
    return length


class CBHG(object):
    """CBHG module from the Tacotron paper.

    It consists of a convolutional stack plus a bidirectional RNN"""
    def __init__(self,
                 input_channels,
                 conv_bank_size, conv_bank_channels,
                 maxpool_width,
                 conv_stack_filter_width, conv_stack_channels,
                 highway_layers, highway_channels,
                 rnn_cells,
                 output_fc_channels=None,
                 batch_norm=False,
                 input_attenuation=20.0,
                 name='CBHG'):
        """Initialize a CBHG module

        Parameters
        ----------
        input_channels : int
            Number of channels of the input tensor.
        conv_bank_size : int
            Number of sets of filters in the convolutional bank. Respective
            filter widths will range from 1 to this size.
        conv_bank_channels : int
            Number of output channels in each of the filter sets in the bank
        maxpool_width : int
            Width of the maxpool layer. Its stride is kept 1 to preserve the
            time resolution
        conv_stack_filter_width : int
            Filter width of the 2-layer convolution stack
        conv_stack_channels : int
            Number of channels of the first layer of the conv stack. The last
            layer has the same number of channels as the input tensor because
            of the residual connection.
        highway_layers : int
            Number of layers in the highway stack.
        highway_channels : int
            Number of channels in the highway stack
        rnn_cell : int
            Dimensionality of memory and output cells of the bidirectional RNN layer
        name : str
            Module's name"""

        self.input_channels = input_channels
        self.conv_bank_size = conv_bank_size
        self.conv_bank_channels = conv_bank_channels
        self.maxpool_width = maxpool_width
        self.conv_stack_filter_width = conv_stack_filter_width
        self.conv_stack_channels = conv_stack_channels
        self.highway_layers = highway_layers
        self.highway_channels = highway_channels
        self.rnn_cells = rnn_cells
        self.num_attention_units = rnn_cells
        self.attention_size = rnn_cells
        self.output_fc_channels = output_fc_channels
        self.batch_norm = batch_norm
        self.input_attenuation = input_attenuation
        self.name = name

    def _conv_bank(self, input_batch):
        """Construct a filter bank with conv_bank_size sets of filters with
        varying filter widths (e.g. set #3 has filters of with 3, set #4
        filters of width 4 and so on up to conv_bank_size)"""

        with tf.variable_scope('conv_bank'):
            bank = []
            for i in range(self.conv_bank_size):
                filt = tflearn.layers.conv_1d(
                    incoming=input_batch,
                    nb_filter=self.conv_bank_channels,
                    filter_size=(i + 1),
                    padding='same',
                    activation='linear',
                    weights_init=tflearn.initializations.xavier(),
                    name='cb_set_{}'.format(i + 1))

                if self.batch_norm:
                    filt = tflearn.batch_normalization(filt)
                filt = tflearn.activations.relu(filt)
                bank.append(filt)

            # concatenate activations along the channel dimension
            return tf.concat(bank, axis=2)

    def _maxpool(self, input_batch):
        """Maxpool layer with stride 1"""
        return tflearn.layers.max_pool_1d(input_batch,
                                          self.maxpool_width,
                                          strides=1,
                                          padding='same',
                                          name='maxpool')

    def _conv_stack(self, input_batch, output_channels):
        """A simple stack of two convolutional layers"""
        with tf.variable_scope('conv_stack'):
            conv1 = tflearn.layers.conv_1d(
                incoming=input_batch,
                nb_filter=self.conv_stack_channels,
                filter_size=self.conv_stack_filter_width,
                padding='same',
                activation='linear',
                weights_init=tflearn.initializations.xavier(),
                name='cs_layer_1')

            if self.batch_norm:
                conv1 = tflearn.batch_normalization(conv1)
            conv1 = tflearn.activations.relu(conv1)

            conv2 = tflearn.layers.conv_1d(
                incoming=conv1,
                nb_filter=output_channels,
                filter_size=self.conv_stack_filter_width,
                padding='same',
                activation='linear',
                weights_init=tflearn.initializations.xavier(),
                name='cs_layer_2')

            if self.batch_norm:
                conv2 = tflearn.batch_normalization(conv2)
            return conv2

    def _highway_net(self, input_batch):
        """Stacked highway network layers"""
        with tf.variable_scope('highway_net'):
            layer = input_batch
            for i in range(self.highway_layers):
                layer = tflearn.layers.highway_conv_1d(
                    incoming=layer,
                    nb_filter=self.highway_channels,
                    filter_size=1,  # i.e. FC layer
                    strides=1,
                    padding='same',
                    activation='relu',
                    name='hw_layer_{}'.format(i + 1))
            return layer

    def _bidirectional_rnn(self, input_batch, sequence_length):
        """Bidirectional RNN with GRU units. It outputs sequence of the same
        length as its input."""
        with tf.variable_scope('GRU_block'):
            cell_fw = tf.contrib.rnn.GRUCell(self.rnn_cells)
            cell_bw = tf.contrib.rnn.GRUCell(self.rnn_cells)
            if sequence_length is None:
                sequence_length = retrieve_seq_length_op(input_batch)
            self.sequence_length = sequence_length

            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=input_batch / self.input_attenuation,
                sequence_length=self.sequence_length,
                dtype=tf.float32)

            return tf.concat(outputs, axis=2), tf.concat(final_states, axis=1)

    def create_network(self, input_batch, sequence_length):
        """Create a CBHG module with following architecture:

        input -> conv_bank -> maxpool -> conv_stack -> + -> highway -> GRURNN
              |________________________________________^
        """
        with tf.variable_scope(self.name):
            net = self._conv_bank(input_batch)
            net = self._maxpool(net)
            net = self._conv_stack(net, self.input_channels)
            net = self._highway_net(net + input_batch)  # residual connection
            outputs, final_states = self._bidirectional_rnn(net,
                                                            sequence_length)

            if self.output_fc_channels is not None:
                with tf.variable_scope('output_fc_projection'):
                    outputs = tflearn.layers.conv_1d(
                            incoming=outputs,
                            nb_filter=self.output_fc_channels,
                            filter_size=1,
                            padding='same',
                            weights_init=tflearn.initializations.xavier(),
                            name='fc_projection')

            return outputs, final_states
