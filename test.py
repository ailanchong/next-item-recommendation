from keras.preprocessing.text import Tokenizer
import math
import numpy as np  
import pickle
import tensorflow as tf 

# mask = [[[1,2,3]],[[4,5,6]]]
# mask = tf.convert_to_tensor(mask)
# mask = tf.squeeze(mask)
#seq_len = [2,1,3]
#mask = tf.cast(tf.sequence_mask(seq_len,maxlen=4), tf.float32)
#mask = tf.expand_dims(mask,0)
#mask = tf.tile(mask,[2,1,1])
mask = [[[ 1,  1,  0,  0],[ 1,  0,  0,  0],[ 1,  1,  1, 0]],[[ 2,  2,  0,  0],[ 2,  0,  0,  0],[ 2,  2,  2,  0]]]
nums = 3
mask = tf.tile(mask,[1,nums,1])
# mask = tf.reshape(mask,[6,3,-1])
#mask = tf.reshape(mask,[9,-1])
# mask = tf.tile(mask,[2,1])

#query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) 
#mask = tf.tile(mask, [3, 1])

# vec = tf.constant([1, 2, 3, 4])
# multiply = tf.constant([3])
# matrix = tf.tile(vec, multiply)
#matrix = tf.reshape(tf.tile(vec, multiply), [ multiply[0], tf.shape(vec)[0]])
sess = tf.Session()
print(sess.run(mask))
print(mask.shape)


