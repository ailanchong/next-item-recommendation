from keras.preprocessing.text import Tokenizer
import math
import numpy as np  
import pickle

def get_table(word_index, word_counts):
    power = 0.75
    table_size = 1e8
    table = np.zeros(table_size, dtype=np.uint32)

    norm = sum([math.pow(word_counts[i], power) for i in word_counts]) # Normalizing constants
    p = 0 # Cumulative probability
    i = 0
    for word in word_counts:
         index = word_index[word]
         count = word_counts[word]
         p += float(math.pow(count, power)) / norm
         while i < table_size and float(i) / table_size < p:
             table[i] = index
             i += 1
    return table

def neg_sample(table, count, positive):
    indices = np.random.randint(low=0, high=len(table), size=2*count)
    result = []
    for index in indices:
        if(table[index] != positive):
            result.append(table[index])
        if(len(result) == count):
            break
    return result

def split(list_tokenized):
    train = []
    test = []
    seq_lens = []
    for line in list_tokenized:
        train.append(line[:-1])
        seq_lens.append(len(line[:-1]))
        test.append(line[-1])
    return train, test, seq_lens

def shuffle_split(train_x, train_y, seq_lens, ratio=0.1):
    index = np.arange(len(train_x))
    np.random.shuffle(index)
    tempX = []
    tempY = []
    tempLen = []
    for key in index:
        tempX.append(train_x[key])
        tempY.append(train_y[key])
        tempLen.append(seq_lens[key])
    train_x = tempX
    train_y = tempY
    seq_lens = tempLen

    train_num = int(len(train_x)*(1-ratio))
    t_x = train_x[0:train_num]
    t_y = train_y[0:train_num]
    v_x = train_x[train_num:-1]
    v_y = train_y[train_num:-1]
    t_lens = seq_lens[0:train_num]
    v_lens = seq_lens[train_num:-1]
    return t_x, t_y, v_x, v_y, t_lens, v_lens

def main(filepath):
    with open(filepath, "rb") as file_in:
        texts = pickle.load(file_in)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    list_tokenized = tokenizer.texts_to_sequences(texts)
    train, test, seq_lens = split(list_tokenized)
    train_x, train_y, test_x, test_y, train_lens, test_lens = shuffle_split(train, test, seq_lens)
    train_x, train_y, val_x, val_y, train_lens, val_lens = shuffle_split(train_x, train_y, train_lens)
    word_index = tokenizer.word_index
    word_counts = tokenizer.word_counts
    table = get_table(word_index, word_counts)

    with open("./data/info", "w") as file_out:
        file_out.write("the total number of items : {}".format(len(word_index)))

    with open("./data/train_corpus","wb") as file_out:
        pickle.dump([train_x, train_y, val_x, val_y, train_lens, val_lens, table], file_out)

    with open("./data/test_corpus", "wb") as file_out:
        pickle.dump([test_x, test_y, test_lens], file_out)

    with open("./data/word_info", "wb") as file_out:
        pickle.dump([word_index, word_counts], file_out)

