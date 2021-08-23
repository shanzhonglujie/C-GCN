from layers import *
from metrics import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}
        # ---------------------------
        self.input_dim=0
        self.input_dim0=0
        self.train_size=0
        self.train_size0=0
        self.word_size=0
        self.word_size0=0

        self.one_hot=None
        self.one_hot0=None
        # ---------------------------
        self.layers = []
        self.activations = []
        self.inputs = None
        #---------------------------
        self.layers0 = []
        self.activations0 = []
        self.inputs0 = None
        #---------------------------
        self.layers1 = []
        self.activations1 = []
        self.inputs1 = None
        #---------------------------
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build_org(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)

        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # 第一个图
        self.activations.append(self.inputs)
        hidden = self.layers[0](self.activations[-1])
        self.activations.append(hidden)
        hidden = self.layers[1](self.activations[-1])
        self.activations.append(hidden)

        # 第二个图
        self.activations0.append(self.inputs0)
        hidden0 = self.layers0[0](self.activations0[-1])
        self.activations0.append(hidden0)
        hidden0 = self.layers0[1](self.activations0[-1])
        self.activations0.append(hidden0)

        # 去除doc向量和word向量
        test_idx =self.train_size+self.word_size
        doc1=tf.concat([self.activations[-1][:self.train_size],self.activations[-1][test_idx:]],axis=0)#句子向量
        word1_ = self.activations[-1][self.train_size:(self.train_size + self.word_size)]#词向量

        test_idx0=self.train_size0+self.word_size0
        doc2=tf.concat([self.activations0[-1][:self.train_size0],self.activations0[-1][test_idx0:]],axis=0)#句子向量
        word2_ = self.activations0[-1][self.train_size0:(self.train_size0 + self.word_size0)]#词向量

        # 单词拼接
        word1,word2,atten = self.layers1[0](word1_,word2_)#SKNet
        doc_word1 = tf.matmul(self.one_hot, word1)
        doc_word2 = tf.matmul(self.one_hot0, word2)
        c = tf.concat([doc1, doc2, doc_word1, doc_word2], axis=1)
        self.outputs = self.layers1[1](c)  # FC

        #输出attention结果
        self.doc = c
        self.word=word1_

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

# GCN
class GCN(Model):
    def __init__(self, placeholders,input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        # self.rating = placeholders['rating']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build_org()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        self.pred = tf.argmax(self.outputs, 1)
        self.labels = tf.argmax(self.placeholders['labels'], 1)

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            sparse_inputs=True,
                                            featureless=True,
                                            dropout=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

#SKIP_GCN
class SKIP_GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(SKIP_GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        # self.rating = placeholders['rating']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build_org()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        self.pred = tf.argmax(self.outputs, 1)
        self.labels = tf.argmax(self.placeholders['labels'], 1)

    def _build(self):
        self.layers.append(GraphConvolution_skip(input_dim=self.input_dim,
                                                 hidden_dim=FLAGS.hidden1,
                                                 output_dim=self.output_dim,
                                                 placeholders=self.placeholders,
                                                 act=lambda x: x,  #
                                                 dropout=True,
                                                 featureless=True,
                                                 sparse_inputs=True,
                                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

# C_GCN
class C_GCN(Model):
    def __init__(self, placeholders, input_param, **kwargs):
        super(C_GCN, self).__init__(**kwargs)
        self.inputs = placeholders['features']
        self.inputs0 = placeholders['features0']

        self.input_dim = input_param['input_dim']
        self.input_dim0 = input_param['input_dim0']
        self.train_size =input_param['train_size']
        self.train_size0 =input_param['train_size0']
        self.word_size=input_param['word_size']
        self.word_size0=input_param['word_size0']
        self.one_hot = placeholders['one_hot']
        self.one_hot0 = placeholders['one_hot0']
        #拼接
        test_idx =self.train_size+self.word_size
        self.nlabels=tf.concat([placeholders['labels'][:self.train_size],placeholders['labels'][test_idx:]],axis=0)
        self.nlabels_mask = tf.concat([placeholders['labels_mask'][:self.train_size], placeholders['labels_mask'][test_idx:]], axis=0)

        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers1[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.layers1[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # Weight decay loss
        self.loss += masked_softmax_cross_entropy(self.outputs, self.nlabels,self.nlabels_mask)

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.nlabels,self.nlabels_mask)
        self.pred = tf.argmax(self.outputs, 1)
        self.labels = tf.argmax(self.placeholders['labels'], 1)

    def _build(self):
        # 第一个图 --------------------------------------------------------------------
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            featureless=True,
                                            act=lambda x: x,
                                            # act=tf.nn.relu,
                                            sparse_inputs=True,
                                            dropout=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            # act=tf.nn.relu,
                                            dropout=True,
                                            logging=self.logging))

        # 第二个图 --------------------------------------------------------------------
        self.layers0.append(GraphConvolutionForTheme(input_dim=self.input_dim0,
                                                     output_dim=FLAGS.hidden1,
                                                     placeholders=self.placeholders,
                                                     act = lambda x: x,
                                                     # act=tf.nn.relu,
                                                     featureless=True,
                                                     sparse_inputs=True,
                                                     dropout=True,
                                                     logging=self.logging))

        self.layers0.append(GraphConvolutionForTheme(input_dim=FLAGS.hidden1,
                                                     output_dim=FLAGS.hidden2,
                                                     placeholders=self.placeholders,
                                                     act=lambda x: x,
                                                     # act=tf.nn.relu,
                                                     dropout=True,
                                                     logging=self.logging))

        # --------------------------------------------------------------------
        #co-train
        self.layers1.append(SKNet_AttLayer(input_col=FLAGS.hidden2,attention_size=32))
        self.layers1.append(Dense(input_dim=FLAGS.hidden2*4,
                                  output_dim=self.output_dim,
                                  placeholders=self.placeholders,
                                  act=lambda x: x,
                                  dropout=True,
                                  logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)
