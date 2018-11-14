import tensorflow as tf
import numpy as np
from tf_utils import *
class TransRecom:
    def __init__(self, batch_size, maxlen, embed_size, item_size, num_sampled=0, num_blocks=1, num_heads=4,  
                 learning_rate=0.001, dropout_rate=0.0, is_training=False):
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.num_sampled = num_sampled
        self.item_size = item_size
        self.embed_size = embed_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.is_training = is_training
        self.num_blocks = num_blocks
        self.num_heads = num_heads

        self.input_x = tf.placeholder(tf.int32, [self.batch_size, self.maxlen], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [self.batch_size,1], name="input_y")
        self.keys_len = tf.placeholder(tf.int32,[self.batch_size], name="keys_len")
        self.querys_len = tf.placeholder(tf.int32, [self.batch_size], name="querys_len")
        if(self.is_training):
            self.sampled_y = tf.placeholder(tf.int32, [self.batch_size*self.num_sampled,1], name="sampled_y")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.enc = embedding(self.input_x, 
                            vocab_size=self.item_size, 
                            num_units=self.embed_size, 
                            scale=True,
                            scope="item_embed")
        
        self.positive_y = embedding(self.input_y, 
                    vocab_size=self.item_size, 
                    num_units=self.embed_size, 
                    scale=True,
                    scope="item_embed",
                    reuse=True)

        if(self.is_training):
            self.negative_y = embedding(self.sampled_y, 
                        vocab_size=self.item_size, 
                        num_units=self.embed_size, 
                        scale=True,
                        scope="item_embed",
                        reuse=True)
            print(self.negative_y.shape)              
        self.inference()
        if(self.is_training):
            self.cal_loss()
            self.train()

    def inference(self):
        with tf.variable_scope("encoder"):
            self.enc += positional_encoding(self.input_x,
                                num_units=self.embed_size, 
                                zero_pad=False, 
                                scale=False,
                                scope="enc_pe")
            ##dropout ps : why?
            self.enc = tf.layers.dropout(self.enc, 
                            rate=self.dropout_rate, 
                            training=tf.convert_to_tensor(self.is_training))

            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.enc = multihead_attention(queries=self.enc, 
                                                    keys=self.enc,
                                                    querys_len=self.querys_len,
                                                    keys_len=self.querys_len, 
                                                    num_units=self.embed_size, 
                                                    num_heads=self.num_heads, 
                                                    dropout_rate=self.dropout_rate,
                                                    is_training=self.is_training,
                                                    causality=False)
                    
                    ### Feed Forward
                    self.enc = feedforward(self.enc, num_units=[4*self.embed_size, self.embed_size])

        with tf.variable_scope("decoder"):
            if(self.is_training):
                self.neg_enc = tf.tile(self.enc,[1, self.num_sampled, 1])
                self.neg_enc = tf.reshape(self.neg_enc, [self.enc.shape.as_list()[0]*self.num_sampled, self.enc.shape.as_list()[1], -1])
            self.positive_y += positional_encoding(self.input_y,
                                num_units=self.embed_size, 
                                zero_pad=False, 
                                scale=False,
                                scope="dec_pe")
            if(self.is_training):
                self.negative_y += positional_encoding(self.sampled_y,
                                    num_units=self.embed_size, 
                                    zero_pad=False, 
                                    scale=False,
                                    scope="dec_pe")

            self.positive_y = multihead_attention(queries=self.positive_y, 
                                            keys=self.enc,
                                            querys_len=tf.ones(shape=[self.positive_y.shape.as_list()[0]]),
                                            keys_len=self.querys_len, 
                                            num_units=self.embed_size, 
                                            num_heads=self.num_heads, 
                                            dropout_rate=self.dropout_rate,
                                            is_training=self.is_training,
                                            causality=False)
            if(self.is_training):
                print(self.negative_y.shape)                                           
                self.negative_y = multihead_attention(queries=self.negative_y, 
                                                keys=self.neg_enc,
                                                querys_len=tf.ones(shape=[self.negative_y.shape.as_list()[0]]),
                                                keys_len=self.querys_len, 
                                                num_units=self.embed_size, 
                                                num_heads=self.num_heads, 
                                                dropout_rate=self.dropout_rate,
                                                is_training=self.is_training,
                                                causality=False,
                                                reuse=True)

            ### Feed Forward
            self.positive_y = feedforward(self.positive_y, num_units=[4*self.embed_size, self.embed_size])
            if(self.is_training):
                self.negative_y = feedforward(self.negative_y, num_units=[4*self.embed_size, self.embed_size], reuse=True)
            
            self.positive_y = tf.squeeze(self.positive_y)
            if(self.is_training):
                self.negative_y = tf.squeeze(self.negative_y)
            
            self.pos_logits = tf.layers.dense(self.positive_y, 1, name="dense_logit")
            if(self.is_training):
                self.neg_logits = tf.layers.dense(self.negative_y, 1, name="dense_logit", reuse=True)

    def cal_loss(self):
        self.pos_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.pos_logits),logits=self.pos_logits))
        neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.neg_logits),logits=self.neg_logits)
        self.loss = self.pos_loss + tf.reduce_mean(neg_loss)

    def train(self):   
        self.train_op = tf.contrib.layers.optimize_loss(self.loss, global_step=self.global_step, learning_rate=self.learning_rate, optimizer="Adam")
    

def test():
    batch_size = 32
    maxlen = 10
    embed_size = 100
    item_size = 128
    num_sampled = 5

    trans = TransRecom(batch_size=32, maxlen=10, embed_size=100, item_size=128, num_sampled=5, is_training=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            input_x = np.ones([batch_size, maxlen])
            input_y = np.ones([batch_size, 1])
            querys_len = np.ones([batch_size])*5
            sampled_y = np.ones([batch_size*num_sampled, 1]) * 2

            #recurrIntereNet.losses, recurrIntereNet.train_op            
            loss = sess.run(trans.pos_loss,feed_dict={
                                             trans.input_x:input_x,
                                             trans.input_y:input_y,
                                             trans.querys_len:querys_len,
                                             trans.keys_len:querys_len
})
            print(loss)    

if __name__ == "__main__":
    test()
