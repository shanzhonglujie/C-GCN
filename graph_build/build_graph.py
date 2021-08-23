import numpy as np
import pickle as pkl
import scipy.sparse as sp
from utils import loadWord2Vec
from math import log
import datetime
import pandas as pd
from datasketch import MinHash, MinHashLSH
from inits import path_name,dataset

is_data_create=True#是否每次都重新生成数据

Tpmi=2
Thash=0.65
dataset='app'
is_yy=True

file_path='app数据'
dict_path='字典'
vector_path= '初始化词向量'

# ngram分词
def bigram_seg_sentence(data):
    words = []
    for dt in data:
        dt=str(dt).strip()
        ls = []
        for i in range(len(dt) - 1):
            temp = dt[i:i + 2]
            ls.append(temp)
        words.append(ls)
    return words

def bit_product_sum(x, y):
    return sum([item[0] * item[1] for item in zip(x, y)])

def get_data_1():
    df = pd.read_csv(file_path)  # 读取数据集
    shuffle_doc_words_list = []
    shuffle_doc_name_list = []
    l = 0
    for i, row in df.iterrows():
        print(i)
        d = row['app_name_list']
        app_name_list = str(d).split('&')
        shuffle_doc_words_list.append(' '.join(app_name_list))
        shuffle_doc_name_list.append(row['y'])
        l += 1
    return shuffle_doc_words_list,shuffle_doc_name_list


def main():
    # 读取app词典
    df = pd.read_csv(dict_path)
    vocab = []
    for i, row in df.iterrows():
        vocab.append(row[0])
    vocab_size = len(vocab)

    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i

    #-----------------------------读取数据-----------------------------------
    word_embeddings_dim = 64
    label_list=[0,1]
    shuffle_doc_words_list, shuffle_doc_name_list=get_data_1()#逾期
    # shuffle_doc_words_list, shuffle_doc_name_list=get_data_2()#性别

    #长度
    all_size = len(shuffle_doc_words_list)
    test_size = int(0.1 * all_size) - 1
    train_size=all_size-test_size
    val_size = int(0.1 * train_size) - 1
    real_train_size = train_size - val_size

    #适应原程序
    train_ids_str='\n'.join([str(i) for i in range(train_size)])
    f = open(path_name+'/' + dataset + '.train.index', 'w')
    f.write(train_ids_str)
    f.close()
    test_ids_str='\n'.join([str(i) for i in range(train_size,train_size+test_size,1)])
    f = open(path_name+'/' + dataset + '.test.index', 'w')
    f.write(test_ids_str)
    f.close()
    #读取向量
    word_vector_file =vector_path
    _, embd, word_vector_map = loadWord2Vec(word_vector_file)

    #-----------------------------保存数据-----------------------------------训练数据
    if is_data_create==True:
        # word_vector_map = {}
        row_x = []
        col_x = []
        data_x = []
        for i in range(real_train_size):
            doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
            doc_words = shuffle_doc_words_list[i]
            words = doc_words.split(' ')
            doc_len = len(words)
            # 计算句向量
            for word in words:
                if word in word_vector_map:
                    word_vector = word_vector_map[word]
                    doc_vec = doc_vec + np.array(word_vector)

            for j in range(word_embeddings_dim):
                row_x.append(i)
                col_x.append(j)
                # np.random.uniform(-0.25, 0.25)
                data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len
        #onehot
        x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
            real_train_size, word_embeddings_dim))
        #y的onehot
        y = []
        for i in range(real_train_size):
            label =shuffle_doc_name_list[i]
            one_hot = [0 for l in range(len(label_list))]
            label_index = label_list.index(label)
            one_hot[label_index] = 1
            y.append(one_hot)
        y = np.array(y)
        print('y的onehot完成')

        # -----------------------------保存数据-----------------------------------测试数据
        # tx: feature vectors of test docs, no initial features
        row_tx = []
        col_tx = []
        data_tx = []
        for i in range(test_size):
            doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
            doc_words = shuffle_doc_words_list[i + train_size]
            words = doc_words.split()
            doc_len = len(words)
            # 计算句向量
            for word in words:
                if word in word_vector_map:
                    word_vector = word_vector_map[word]
                    doc_vec = doc_vec + np.array(word_vector)

            for j in range(word_embeddings_dim):
                row_tx.append(i)
                col_tx.append(j)
                # np.random.uniform(-0.25, 0.25)
                data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

        # tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
        tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                           shape=(test_size, word_embeddings_dim))

        ty = []
        for i in range(test_size):
            label = shuffle_doc_name_list[i + train_size]
            one_hot = [0 for l in range(len(label_list))]
            label_index = label_list.index(label)
            one_hot[label_index] = 1
            ty.append(one_hot)
        ty = np.array(ty)
        print('ty的onehot完成')

    # -----------------------------保存数据-----------------------------------测试数据+词汇
    # allx: the the feature vectors of both labeled and unlabeled training instances
    # (a superset of x)
    # unlabeled training instances -> words
    if is_data_create==True:
        word_vectors = np.random.uniform(-0.01, 0.01,
                                         (vocab_size, word_embeddings_dim))

        for i in range(len(vocab)):
            word = vocab[i]
            if word in word_vector_map:
                vector = word_vector_map[word]
                word_vectors[i] = vector

        row_allx = []
        col_allx = []
        data_allx = []
        train_vec=[]
        for i in range(train_size):
            doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
            doc_words = shuffle_doc_words_list[i]
            words = doc_words.split(' ')
            doc_len = len(words)
            for word in words:
                if word in word_vector_map:
                    word_vector = word_vector_map[word]
                    doc_vec = doc_vec + np.array(word_vector)

            for j in range(word_embeddings_dim):
                row_allx.append(int(i))
                col_allx.append(j)
                # np.random.uniform(-0.25, 0.25)
                data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len

            train_vec.append(doc_vec/doc_len)
        for i in range(vocab_size):
            for j in range(word_embeddings_dim):
                row_allx.append(int(i + train_size))
                col_allx.append(j)
                data_allx.append(word_vectors.item((i, j)))

        row_allx = np.array(row_allx)
        col_allx = np.array(col_allx)
        data_allx = np.array(data_allx)

        allx = sp.csr_matrix(
            (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

        ally = []
        for i in range(train_size):
            label = shuffle_doc_name_list[i]
            one_hot = [0 for l in range(len(label_list))]
            label_index = label_list.index(label)
            one_hot[label_index] = 1
            ally.append(one_hot)

        for i in range(vocab_size):
            one_hot = [0 for l in range(len(label_list))]
            ally.append(one_hot)

        ally = np.array(ally)
        print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    #========================================= 初始化
    # 通过词对获得权重
    row = []
    col = []
    weight = []

    windows = []
    labels=[]
    dict_keys=word_id_map.keys()#字典里的id
    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = doc_words.split(' ')
        windows.append(words)
        labels.append(shuffle_doc_name_list[i])

    #判断是否是同一类别的边
    def get_cut_weight(labels,idx_i,idx_j):
        label_i=labels[idx_i]
        label_j=labels[idx_j]
        return label_i==label_j


    # ========================================= onehot
    one_hot = np.empty([train_size + test_size, vocab_size], dtype=int)
    start=datetime.datetime.now()
    for i, words in enumerate(windows):
        for word in words:
            if word not in dict_keys:
                continue
            j = word_id_map[word]
            one_hot[i,j]=1
    end=datetime.datetime.now()
    print('创建onehot矩阵,执行了%f秒' % ((end - start).seconds))

    #========================================= y-y
    if is_yy == True:
        start=datetime.datetime.now()
        b_num1=0

        hash_array = []
        win_len=len(windows)
        # Create MinHash objects
        for i in range(win_len):
            hash_array.append(MinHash(num_perm=200))

        for i, d in enumerate(windows):
            for word in d:
                hash_array[i].update(str(word).encode('utf8'))

        lsh = MinHashLSH(threshold=Thash, num_perm=200)
        for i, hash_s in enumerate(hash_array):
            lsh.insert(str(i), hash_s)

        for k, hash in enumerate(hash_array):
            if k==win_len-1:
                break
            result = lsh.query(hash_array[k])
            if len(result)>1:
                # print(result)
                result_num=[ int(i) for i in result]
                for n1 in result_num:
                    for n2 in result_num:
                        if n1 != n2 and get_cut_weight(labels,n1,n2)==True:
                            b_num1 += 1
                            if n1 < train_size:
                                row.append(n1)
                            else:
                                row.append(n1 + vocab_size)
                            if n2 < train_size:
                                col.append(n2)
                            else:
                                col.append(n2 + vocab_size)
                            weight.append(1)
        end=datetime.datetime.now()
        print('计算y与y相似度，增加了%d条边，执行了%f秒' % (b_num1,(end - start).seconds))

    #
    # ========================================= x-x
    '''
    Doc word heterogeneous graph
    '''
    start=datetime.datetime.now()
    word_window_freq = {}
    #获得p 每个app的统计数量
    for window in windows:
        for word in window:
            if word not in vocab:
                continue
            if word in word_window_freq:
                word_window_freq[word] += 1
            else:
                word_window_freq[word] = 1
    print('单个app出现频率计算结束')

    word_pair_count = {}
    for window in windows:
        length=len(window)
        for i in range(1, length):
            for j in range(i, length):
                if i==j:
                    continue
                word_i = window[i]
                word_j = window[j]
                if word_i not in dict_keys or word_j not in dict_keys:
                    continue
                word_i_id = word_id_map[word_i]#获得位置索引
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders 反方向
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
    print('两个app同时出现频率计算结束')

    num_window = len(windows)
    b_num2=0
    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        # has_loan_app=is_loan_app(vocab[i], vocab[j])
        key = str(i) + ',' + str(j)
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
        # print(pmi)
        if pmi <Tpmi:
            continue

        b_num2+=1
        row.append(train_size + i)
        col.append(train_size + j)
        weight.append(pmi)
    end=datetime.datetime.now()
    print('计算pmi结束，增加了%d条边，执行了%f秒' % (b_num2,(end - start).seconds))


    #========================================= x y
    word_doc_list = {}
    start=datetime.datetime.now()
    for i,words in enumerate(windows):
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)

    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    b_num3=0
    for i, words in enumerate(windows):
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            if word not in dict_keys:
                continue

            j = word_id_map[word]
            freq = 1#常规图
            if i < train_size:
                row.append(i)
            else:
                row.append(i + vocab_size)
            col.append(train_size + j)
            idf = log(1.0 * len(shuffle_doc_words_list) /
                      word_doc_freq[vocab[j]])
            b_num3+=1
            weight.append(freq * idf)
            doc_word_set.add(word)
    end=datetime.datetime.now()
    print('计算tfidf结束,增加了%d条边,执行了%f秒' % (b_num3,(end - start).seconds))
    node_size = train_size + vocab_size + test_size
    # -----------------------------------------------------------------
    adj = sp.csr_matrix(
        (weight, (row, col)), shape=(node_size, node_size))

    if is_data_create==True:
        # dump objects
        f = open(path_name+"/ind.{}.x".format(dataset), 'wb')
        pkl.dump(x, f)
        f.close()

        f = open(path_name+"/ind.{}.y".format(dataset), 'wb')
        pkl.dump(y, f)
        f.close()

        f = open(path_name+"/ind."+dataset+".tx", 'wb')
        pkl.dump(tx, f)
        f.close()

        f = open(path_name+"/ind."+dataset+".ty", 'wb')
        pkl.dump(ty, f)
        f.close()

        f = open(path_name+"/ind."+dataset+".allx", 'wb')
        pkl.dump(allx, f)
        f.close()

        f = open(path_name+"/ind."+dataset+".ally", 'wb')
        pkl.dump(ally, f)
        f.close()

        f = open(path_name+"/ind."+dataset+".onehot", 'wb')
        pkl.dump(one_hot, f)
        f.close()
    f = open(path_name+"/ind."+dataset+".adj", 'wb')
    pkl.dump(adj, f)
    f.close()

    print('--ok--')