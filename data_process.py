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
    for line in list_tokenized:
        train.append(line[:-1])
        test.append(line[-1])
    return train, test



def main(filepath):
    with open(filepath, "rb") as file_in:
        texts = pickle.load(file_in)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    list_tokenized = tokenizer.texts_to_sequences(texts)
    train, test = split(list_tokenized)

    word_index = tokenizer.word_index
    word_counts = tokenizer.word_counts
    table = get_table(word_index, word_counts)

    with open("./data/info", "w") as file_out:
        file_out.write("the total number of items : {}".format(len(word_index)))

    with open("./data/train_corpus","wb") as file_out:
        pickle.dump([train, test, table], file_out)

    with open("./data/word_info", "wb") as file_out:
        pickle.dump([word_index, word_counts], file_out)

