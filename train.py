from hyperparams import hyperparams as hp
import tensorflow as tf 
from TransRecom import TransRecom
import numpy as np 
import pickle

def zero_padding(sentence_list, maxlen):
    result = np.zeros((len(sentence_list), maxlen), dtype=np.int)
    for i, row in enumerate(sentence_list):
        for j, val in enumerate(row):
            result[i][j] = val
    return result

def neg_sample(table, count, positive):
    indices = np.random.randint(low=0, high=len(table), size=2*count)
    result = []
    for index in indices:
        if(table[index] != positive):
            result.append(table[index])
        if(len(result) == count):
            break
    return result

def generate_batch(trainX, trainY, seq_lens, batch_size, table):
    batch_x = []
    batch_y = []
    batch_sample = []
    batch_lens = []
    curr_num = 0
    while 1:
        totalnum = len(trainX)
        index = np.arange(totalnum)
        np.random.shuffle(index)
        tempX = []
        tempY = []
        tempLen = []
        for key in index:
            tempX.append(trainX[key])
            tempY.append(trainY[key])
            tempLen.append(seq_lens[key])
        trainX = tempX
        trainY = tempY
        seq_lens = tempLen
        for i in range(totalnum):
            curr_num += 1
            batch_x.append(trainX[i])
            batch_y.append(trainY[i])
            batch_sample.extend(neg_sample(table, hp.num_sampled, trainY[i]))
            batch_lens.append(seq_lens[i])
            if curr_num == batch_size:
                yield batch_x, np.asarray(batch_y), np.asarray(batch_sample), np.asarray(batch_lens)
                batch_x = []
                batch_y = []
                batch_sample = []
                batch_lens = []
                curr_num = 0




def main():
    with open("./data/train_corpus", "rb") as file_in:
        train_x, train_y, val_x, val_y, train_lens, val_lens, table = pickle.load(file_in)
    val_x = zero_padding(val_x, hp.maxlen)

    transRecom = TransRecom(hp.batch_size, hp.maxlen, hp.embed_size,
                            hp.item_size, num_sampled = hp.num_sampled,
                            is_training=hp.is_training)
    saver = tf.train.Saver()
    step = tf.Variable(0,trainable=False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_gendata = generate_batch(train_x, train_y, train_lens, hp.batch_size, table)
       
        for i in range(hp.num_epochs):
            for j in range(len(train_x) // hp.batch_size):
                batch_x, batch_y, batch_sample, batch_lens = next(train_gendata)
                batch_x = zero_padding(batch_x, hp.maxlen)
                _, step_num, loss = sess.run([transRecom.train_op, step, transRecom.loss], feed_dict={
                    transRecom.input_x:batch_x,
                    transRecom.input_y:batch_y,
                    transRecom.sampled_y:batch_sample,
                    transRecom.querys_len:batch_lens,
                    transRecom.keys_len:batch_lens
                })
                print("batch_step: %s, train_rmse: %s" %(step_num, loss))

            test_loss = sess.run([transRecom.pos_loss], feed_dict={
                    transRecom.input_x:val_x,
                    transRecom.input_y:val_y,
                    transRecom.querys_len:val_lens,
                    transRecom.keys_len:val_lens
            })
            print("iter_step: %s, test_rmse: %s" %(i, test_loss))
            #模型保存
            saver.save(sess, "./model/transRecom.ckpt")

if __name__ == "__main__":
    main()



