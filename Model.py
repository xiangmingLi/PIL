from functions import *
import tensorflow.contrib.layers as layers
from base_model import Basemodel
class Baseline(Basemodel):
    """
     Baseline_V2: Compared to Baseline_V1, the main difference is that Baseline_V2 use element-wise adding to perform
     fusion before the softmax layer.
    """
    def MFB_Soft_Attention(self, conv_f, f_q, scale, name_scope, mfb_dim=config.MFB_DIM, mfb_out=config.MFB_OUT_DIM,
                           mfb_factor_num=config.MFB_FACTOR_NUM, f_q_dim=config.QUESTION_RNN_DIM):
        with tf.variable_scope(name_scope):
            F_proj = self.conv2d(conv_f,mfb_dim,kernel_size=[1,1],activation=None,name='F_proj') # N, 7,7,5000
            F = tf.reshape(conv_f, [-1, scale * scale, self.V])
            h_proj = self.dense(f_q,mfb_dim,activation=None,name='q_proj')# N,5000
            h = tf.expand_dims(h_proj, axis=1) # N,1,5000
            h = tf.expand_dims(h, axis=1) # N,1,15000
            h_tile = tf.tile(h, [1, scale, scale, 1])

            F_j = self.mfb_matrix_fusion(F_proj, h_tile, scale * scale, mfb_out, mfb_factor_num)  # N,49,1000
            F_j = tf.reshape(F_j,[-1,scale,scale,mfb_out]) # N,7,7,1000
            F_j_conv=self.conv2d(F_j,512,(1,1),activation=tf.nn.relu) # N,7,7,512
            att_logits=self.conv2d(F_j_conv,filters=1,kernel_size=(1,1),activation=None,use_bias=False,name='att_logits') # N,7,7,1
            att_logits = tf.reshape(att_logits, [-1, scale * scale])  # N, 49
            att = tf.nn.softmax(att_logits)  # Attention Values, shape [N,49]
            F_att = F * tf.expand_dims(att, 2)
            f_v = tf.reduce_sum(F_att, axis=1, name='attention_feature')
            return f_v, att

    def Soft_Attention(self, conv_f, f_q):
        F = conv_f # N, 196, 2048
        F = tf.reshape(F, [-1, self.V])  # Reshape the conv_feature[N*49, 2048]
        F_p=self.dense(F,self.H,activation=None,name='img_proj')
        F_p = tf.reshape(F_p, [-1, self.R, self.H])  # Reshape F_p [N,49, 1024(VP)]

        h = self.dense(f_q,self.H,activation=None,name='q_proj')
        h = tf.expand_dims(h, 1)  # Expand projected question feature to [N,1,QP]
        h = tf.tile(h, [1, self.R, 1])  # Perform tile operation, [N,196,H]
        # Perform Fusion Operation
        F_j = tf.add(F_p, h)  # Shape [N,49,1024(VP,QP)]
        F_j=tf.nn.tanh(F_j)
        if self.dropout:
            F_j = tf.nn.dropout(F_j, keep_prob=self.keep_prob)

        # Perform Softmax and Obtain Attention Values
        F_j = tf.reshape(F_j, [-1, self.H])  # Shape [N*49, 1024(VP or QP)]
        att_logits=self.dense(F_j,1,activation=None,use_bias=False,name='attention_logits')
        att_logits=tf.reshape(att_logits,[-1,self.R]) # N,196
        att = tf.nn.softmax(att_logits)  # Attention Values, shape [N,196]
        att = tf.expand_dims(att,2) # N,196,1
        f_v=tf.reduce_sum(conv_f*att,axis=1)
        return f_v, att

    def single_vqa_channel(self,img_vec,f_q,reuse = False):
        # Project image feature matrix
        with tf.variable_scope('vqa_scope',reuse = reuse):
            conv_f = tf.transpose(img_vec, [0, 2, 3, 1])  # Transpose the conv_feature into tensorflow format
            # conv_f = tf.reshape(conv_f, [-1, self.R, self.V])  # N,196,V
            # if self.dropout:
            #     conv_f = tf.nn.dropout(conv_f, keep_prob=self.keep_prob)
            with tf.variable_scope('attention_operation'):
                f_v, att_values = self.MFB_Soft_Attention(conv_f,f_q,14,'attention')
            f_j = self.mfb_vector_fusion(f_v,f_q,'joint_fusion')
            ff=self.dense(f_j,config.FF_DIM,name='forward_layer')
            logits = self.dense(ff,self.A,activation=None, name='prediction_layer')
            return logits,ff,f_v,att_values

    def build_vqa_module(self):
        # Initialize input tensors: image_features,
        #  question_input, answer_vector,
        # setence_length, dropout rate
        self.img_vec = tf.placeholder(tf.float32, [None, 2048, 14, 14], 'conv_features1')  # input Res Features
        self.img_vec2 = tf.placeholder(tf.float32, [None, 2048, 14, 14], 'conv_features2')  # input Res Features
        self.q_input = tf.placeholder(tf.float32, [None, self.q_max, self.E])  # Question Input, consists of word index
        self.a_input = tf.placeholder(tf.float32,[None,self.E])
        self.a_input2 = tf.placeholder(tf.float32, [None, self.E])
        self.ans_space_score = tf.placeholder(tf.float32, [None])
        self.ans1 = tf.placeholder(tf.float32, [None, self.A],'ans1')  # Answer Vector Input.
        self.ans2 = tf.placeholder(tf.float32, [None, self.A],'ans2')  # Answer Vector Input.
        self.seqlen = tf.placeholder(tf.int32, [None])  # Sentence lengths
        self.is_training = tf.placeholder(tf.bool)  # Is_training parameter

        # Obtain question feature
        batch_size = tf.shape(self.q_input)[0]
        self.f_q = self.build_lstm_modules(self.q_input, self.seqlen, batch_size)

        self.logits1,self.ff21,self.fv_1,self.att1_values = self.single_vqa_channel(self.img_vec,self.f_q,False)
        self.logits2,self.ff22,self.fv_2,self.att2_values = self.single_vqa_channel(self.img_vec2, self.f_q,True)

        self.predict1 = tf.argmax(tf.nn.softmax(self.logits1), axis=1)
        self.predict2 = tf.argmax(tf.nn.softmax(self.logits2), axis=1)

        # Obtain different loss.
        with tf.name_scope('cross_entrophy1'):
            self.sigmoid_cross_entrophy1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits1,
                                                                                                 labels=self.ans1))*config.VQA_ANS_OUTPUT  # Averaged cross entrophy loss
            self.softmax_cross_entrophy1 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.logits1, labels=self.ans1))
        with tf.name_scope('cross_entrophy2'):
            self.sigmoid_cross_entrophy2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits2,
                                                                                                 labels=self.ans2))*config.VQA_ANS_OUTPUT  # Averaged cross entrophy loss
            self.softmax_cross_entrophy2 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.logits2, labels=self.ans2))
        self.fa1 = self.dense(self.ff21,self.E,activation=None,name='ans_feature')
        self.fa2 = self.dense(self.ff22, self.E, activation=None, name='ans_feature',reuse= True)
        distance = tf.sqrt(tf.reduce_sum(tf.pow(self.a_input - self.fa1, 2), 1, keep_dims=True))
        self.distance = tf.reduce_mean(distance)
        distance2 = tf.sqrt(tf.reduce_sum(tf.pow(self.a_input - self.fa2, 2), 1, keep_dims=True))
        self.distance2 = tf.reduce_mean(distance2)
        dapn = tf.sqrt(tf.reduce_sum(tf.pow(self.a_input - self.a_input2, 2), 1, keep_dims=True))
        self.dapn = tf.reduce_mean(dapn)
        tmp3 = tf.maximum((dapn + distance - distance2), 0)
        self.distance_loss = tf.reduce_mean(tmp3)
        self.loss1 = self.sigmoid_cross_entrophy1
        self.loss2 = self.sigmoid_cross_entrophy2
        self.overloss = (self.loss1 + self.loss2)/2  + 0.01*self.distance_loss

        correct_prediction1 = tf.equal(self.predict1, tf.argmax(self.ans1, 1))
        correct_prediction2 = tf.equal(self.predict2, tf.argmax(self.ans2, 1))
        with tf.name_scope('accuracy1'):
            accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
        with tf.name_scope('accuracy2'):
            accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
        self.accuracy1 = accuracy1
        self.accuracy2 = accuracy2

        dpn = tf.sqrt(tf.reduce_sum(tf.pow(self.fa1 - self.fa2, 2), 1, keep_dims=True))
        self.dpn = tf.reduce_mean(dpn)



    def build_lstm_modules(self, word_embs, seqlen, batch_size):

        """
        Build one layer lstm, and obtain question feature f_q
        :param x_ids:
        :param seqlen:
        :param batch_size:
        :return:
        """
        x = word_embs

        with tf.variable_scope('question_module'):
            lstm_cell = tf.nn.rnn_cell.GRUCell(self.H)
            _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

            if self.dropout:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.7)
            outputs, states = tf.nn.dynamic_rnn(
                cell=lstm_cell,
                inputs=x,
                dtype=tf.float32,
                sequence_length=seqlen,
                initial_state=_init_state
            )
            return states
