from metrics import classify_evalue
import time
from utils import *
from models import GCN,C_GCN,SKIP_GCN

#Set cpu
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True

# Set random seed
seed = 234
np.random.seed(seed)
tf.set_random_seed(seed)
'''
论文算法
'''
# Settings
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

flags = tf.app.flags
FLAGS = flags.FLAGS

# 'gcn', 'skip_gcn', 'cgcn'
flags.DEFINE_string('model', 'cgcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
flags.DEFINE_integer('epochs',200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 200 , 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 50, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')  # 5e-4
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
if FLAGS.model == 'cgcn':
    dataset = "data"# our graph building data
else:
    dataset = "data_textgcn" # graph building data of textgcn

flags.DEFINE_string('dataset', dataset, 'Dataset string.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size,word_size,one_hot = load_corpus('app',path_name=dataset)
adj0, features0, y_train0, y_val0, y_test0, train_mask0, val_mask0, test_mask0, train_size0, test_size0,word_size0,one_hot0= load_corpus('theme')

# featureless
features = sp.identity(features.shape[0])
features0 = sp.identity(features0.shape[0])
# Some preprocessing
features = preprocess_features(features)
features0 = preprocess_features(features0)
if FLAGS.model == 'cgcn':#同时对常规图和次主题图进行训练
    support = [preprocess_adj(adj)]
    support0 = [preprocess_adj(adj0)]
    num_supports = 1
    model_func = C_GCN
elif FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    support0 = []
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'skip_gcn':
    support = [preprocess_adj(adj)]
    support0 = []
    num_supports = 1
    model_func = SKIP_GCN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(1)],
    'support0': [tf.sparse_placeholder(tf.float32) for _ in range(1)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'features0': tf.sparse_placeholder(tf.float32, shape=tf.constant(features0[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels0': tf.placeholder(tf.float32, shape=(None, y_train0.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'labels_mask0': tf.placeholder(tf.int32),
    'one_hot': tf.placeholder(tf.float32, shape=(train_size + test_size, word_size)),
    'one_hot0': tf.placeholder(tf.float32, shape=(train_size0 + test_size0, word_size0)),
    'dropout': tf.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': tf.placeholder(tf.int32),
    'num_features_nonzero0': tf.placeholder(tf.int32),
}
# Create model
input_dict={
    'input_dim':features[2][1],
    'input_dim0':features0[2][1],
    'train_size': train_size,
    'train_size0': train_size0,
    'word_size': word_size,
    'word_size0': word_size0,
}
if FLAGS.model == 'cgcn':
    model = model_func(placeholders, input_param=input_dict, logging=True)
else:
    model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session(config=config)

# Define model evaluation function
def evaluate(features,support,labels, mask, one_hot, features0,support0, labels0, mask0, one_hot0,placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(
        features,support,labels, mask,one_hot, features0,support0, labels0, mask0,one_hot0,placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test)

# Init variables
sess.run(tf.global_variables_initializer())
cost_val = []

# Train model
test_acc_lst=[]
for epoch in range(FLAGS.epochs):
    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(
        features, support,y_train, train_mask, one_hot,features0, support0, y_train0, train_mask0, one_hot0,placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    if FLAGS.model == 'cgcn':
        outs = sess.run([model.opt_op, model.loss, model.accuracy, model.doc,model.word], feed_dict=feed_dict)
    else:
        outs = sess.run([model.opt_op, model.loss, model.accuracy, model.layers[0].embedding], feed_dict=feed_dict)

    # Validation
    cost, acc, pred, labels, duration = evaluate(features, support, y_val, val_mask, one_hot, features0, support0,y_val0, val_mask0, one_hot0, placeholders)
    cost_val.append(cost)

    # Testing
    test_cost, test_acc, _, _, test_duration = evaluate(
        features, support, y_test, test_mask,one_hot,features0,support0,y_test0, test_mask0,one_hot0,placeholders)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc),
          "Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc))

    if epoch>5:
        test_acc_lst.append(test_acc)
    #记录结果
    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

#---------------------------开始评价----------------------------
# Testing
test_cost, test_acc, pred, labels, test_duration = evaluate(
    features, support, y_test, test_mask,one_hot,features0,support0,y_test0, test_mask0,one_hot0,placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

test_pred = []
test_labels = []

if FLAGS.model == 'cgcn':
    test_idx = train_size + word_size
    nlabels = np.hstack((labels[:train_size], labels[test_idx:]))
    for i in range(len(pred)):
        if i > train_size:
            test_pred.append(pred[i])
            test_labels.append(nlabels[i])
else:
    for i in range(len(pred)):
        if i>train_size+word_size:
            test_pred.append(pred[i])
            test_labels.append(labels[i])

print('max value is '+str(np.max(test_acc_lst)))
print('min value is '+str(np.min(test_acc_lst)))
acc_r =classify_evalue(test_labels, test_pred, 'acc')
f1_r =classify_evalue(test_labels, test_pred, 'f1')
roc_auc_r =classify_evalue(test_labels, test_pred, 'roc_auc')
print(acc_r)
print(f1_r)
print(roc_auc_r)

#----------------------------vector----------------------------
print('embeddings:')
# doc and word embeddings
doc_embeddings_name = 'vector/app_doc_vectors.txt'
word_embeddings_name ='vector/app_word_vectors.txt'

if FLAGS.model == 'cgcn':
    word_embeddings = outs[4]
    train_doc_embeddings = outs[3][:train_size]  # include val docs
    test_doc_embeddings = outs[3][train_size:]
else:
    word_embeddings = outs[3][train_size: adj.shape[0] - test_size]
    train_doc_embeddings = outs[3][:train_size]
    test_doc_embeddings = outs[3][adj.shape[0] - test_size:]

write_vector(word_embeddings_name, doc_embeddings_name, word_embeddings, train_doc_embeddings, test_doc_embeddings,train_size,test_size,word_size)


