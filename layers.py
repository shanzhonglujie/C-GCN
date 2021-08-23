from inits import *
from utils import *

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()


    def _call(self, inputs):
        x = inputs
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)
        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)
        # bias
        if self.bias:
            output += self.vars['bias']
        self.embedding = output
        return self.act(output)

#skip-GCN
class GraphConvolution_skip(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, hidden_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution_skip, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs #不能没有
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, hidden_dim],
                                                        name='weights_' + str(i))
                self.vars['weights_2' + str(i)] = glorot([hidden_dim, output_dim],
                                                         name='weights_2' + str(i))
                self.vars['weights_skip_' + str(i)] = glorot([input_dim, output_dim],
                                                             name='weights_skip_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # dropout
        x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        # convolve
        supports_ = list()
        for i in range(len(self.support)):
            pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports_.append(support)
        hidden = tf.nn.relu(tf.add_n(supports_))

        hidden = tf.nn.dropout(hidden, 1 - self.dropout)
        # convolve
        supports = list()
        for i in range(len(self.support)):
            pre_sup = dot(hidden, self.vars['weights_2' + str(i)],False)
            pre_sup_skip = dot(x, self.vars['weights_skip_' + str(i)],True)
            support = dot(self.support[i], pre_sup, sparse=True)+pre_sup_skip
            supports.append(support)
        output = tf.add_n(supports)
        # bias
        if self.bias:
            output += self.vars['bias']
        self.embedding = output
        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)
        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)
        # bias
        if self.bias:
            output += self.vars['bias']
        self.embedding = output
        return self.act(output)

#次主题-同GCNConv
class GraphConvolutionForTheme(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolutionForTheme, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support0']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero0']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)
        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)
        # bias
        if self.bias:
            output += self.vars['bias']
        self.embedding = output
        return self.act(output)

#sknet注意力机制
class SKNet_AttLayer(Layer):
    def __init__(self, input_col, attention_size, dropout=0.6, var_name='0', **kwargs):
        super(SKNet_AttLayer, self).__init__(**kwargs)
        self.attention_size = attention_size
        self.dropout=dropout
        self.w_omega = glorot([input_col, self.attention_size],name='w_omega'+var_name)
        self.b_omega = glorot([1,self.attention_size], name='b_omega' + var_name)
        self.u_omega = glorot([self.attention_size,1], name='u_omega' + var_name)

    def __call__(self, input1, input2):
        input1 = tf.nn.dropout(input1, 1 - self.dropout)
        input2 = tf.nn.dropout(input2, 1 - self.dropout)
        c = tf.shape(input1)[0]
        r = tf.shape(input2)[0]
        inputs1_ = tf.expand_dims(input1, -1)
        inputs1 = tf.transpose(inputs1_, [0, 2, 1])
        inputs2_ = tf.expand_dims(input2, -1)
        inputs2 = tf.transpose(inputs2_, [0, 2, 1])
        inputs = tf.concat([inputs1, inputs2], axis=0)
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        with tf.variable_scope('v'):
            v = tf.nn.tanh(tf.tensordot(inputs, self.w_omega, axes=1) + self.b_omega)
        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, self.u_omega, axes=1, name='vu')  # (B,T) shape
        vu = tf.squeeze(vu) # 权重
        vu1, vu2 = vu[:c], vu[c:]
        alphas1 = tf.nn.softmax(vu1, name='alphas')  # (B,T) shape
        w1 = tf.expand_dims(alphas1, axis=-1)
        alphas2 = tf.nn.softmax(vu2, name='alphas')  # (B,T) shape
        w2 = tf.expand_dims(alphas2, axis=-1)
        alphas=tf.concat([w1,w2],axis=0)
        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output1 = tf.reduce_sum(inputs1 * tf.expand_dims(w1, -1), 1)
        output2 = tf.reduce_sum(inputs2 * tf.expand_dims(w2, -1), 1)
        self.embedding =alphas
        return output1,output2,alphas