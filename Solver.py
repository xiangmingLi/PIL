__author__ = 'antony'
import config
import tensorflow as tf
import numpy as np
import time,math,utils,json,os
import shutil,sys
from visulize_attention import *
from new_data_loader import VQADataLoader
from pairwaise_dataloader import VQADataLoader_pair
from utils import visualize_failures
sys.path.append(config.VQA_TOOLS_PATH)
sys.path.append(config.VQA_EVAL_TOOLS_PATH)

from vqaTools.vqa import VQA
from vqaEvaluation.vqaEval import VQAEval

from scipy import *

class Solver(object):

    def __init__(self,model):

        self.model=model
        self.global_epoch_step=0
        self.lr=tf.placeholder(tf.float32)
        self.global_step=tf.Variable(0,name='global_step',trainable=False)
        with tf.variable_scope(config.VQA_SCOPE):
            self.vqa_optimizer=tf.train.AdamOptimizer(self.lr)


    def build_train_ops(self):
        learning_rate=config.VQA_LR
        def _learning_rate_decay_fn(learning_rate,global_step):
            return tf.train.exponential_decay(learning_rate=learning_rate,global_step=global_step,
                                                          decay_steps=config.NUM_STEPS_PER_DECAY,
                                                       decay_rate=config.DECAY_RATE,
                                                       staircase=True)
        learning_rate_decay=_learning_rate_decay_fn

        decay_lr = tf.train.exponential_decay(learning_rate=learning_rate,global_step=self.global_step,
                                                          decay_steps=config.NUM_STEPS_PER_DECAY,
                                                       decay_rate=config.DECAY_RATE,
                                                       staircase=True)

        with tf.variable_scope('optimizer',reuse=tf.AUTO_REUSE):
            self.vqa_optimizer=tf.train.AdamOptimizer(
                learning_rate=config.VQA_LR
            )

        train_ops=tf.contrib.layers.optimize_loss(
            loss=self.overall_loss,
            global_step=self.global_step,
            learning_rate=learning_rate,
            optimizer=self.vqa_optimizer,
            clip_gradients=config.CLIP_GRADIENT,
            learning_rate_decay_fn=learning_rate_decay
        )
        return train_ops,decay_lr

    def save_vqa_model(self,sess,scope,step,update=False):
        path=config.FOLDER_NAME+'/models/'
        saver=tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES,scope=scope))
        print 'saving model to %s'%path
        if update:
            saver.save(sess,path+config.MODEL_SAVE_NAME)
        else:
            saver.save(sess,path+config.MODEL_SAVE_NAME,step)


    def idx2answers(self,predicts):
        return [self.dataloader.idx2ans[str(x)] for x in predicts]

    def load_vqa_model(self,sess,scope):
        path = config.FOLDER_NAME + '/models/'
        saver=tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES,scope=scope))
        saver.restore(sess,path+config.MODEL_LOAD_NAME)

    def train(self):
        #################### Basic Operations#########################
        if config.POSITIVE_ONLY:
            self.overall_loss=self.model.loss1
            cross_entrophy=self.model.sigmoid_cross_entrophy1
            accuracy=self.model.accuracy1
            vqa_predict1=self.model.predict1
            af_distance = self.model.distance
        else:
            self.overall_loss = self.model.overloss
            cross_entrophy = self.model.sigmoid_cross_entrophy1+self.model.sigmoid_cross_entrophy2
            accuracy = (self.model.accuracy1 + self.model.accuracy2)/2
            vqa_predict1 = self.model.predict1
            vqa_predict2 = self.model.predict2
            af_distance = self.model.distance
            vqa_distance = self.model.distance2
            distance_loss = self.model.distance_loss
            pn_distance = self.model.dpn
            apn_distance = self.model.dapn

        vqa_train_op,vqa_decay_lr=self.build_train_ops()
        #################### Summary Setting #########################
        val_average_loss=tf.placeholder('float')
        val_average_accuracy=tf.placeholder('float')

        train_ol_summary=tf.summary.scalar('train_overall_loss',self.overall_loss)
        train_ce_summary=tf.summary.scalar('train_cross_entrophy',cross_entrophy)
        train_acc_summary=tf.summary.scalar('accuracy',accuracy)

        val_average_l_summary=tf.summary.scalar('val_ave_cross_entrophy',val_average_loss)
        val_average_acc_summary=tf.summary.scalar('val_average_accuracy',val_average_accuracy)

        #################### Session Setting#########################
        sess=tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(tf.global_variables_initializer())

        train_writer=tf.summary.FileWriter(config.log_path+'train',sess.graph)
        valid_writer=tf.summary.FileWriter(config.log_path+'valid')

        if config.USE_PRE_VQA_MODEL:
            self.load_vqa_model(sess,config.VQA_SCOPE)
        #################### Train Step Function#########################

        dataloader=VQADataLoader(batchsize=config.BATCH_SIZE,mode='train')
        dataloader_pairs = VQADataLoader_pair(batchsize=config.BATCH_SIZE, mode='train')
        start_t = time.time()
        best_result = [0.0, 0]
        val_counter = 0
        for i in range(config.MAX_TRAIN_STEP):
            if config.POSITIVE_ONLY:
                #data_time=time.time()
                q_strs, q_word_vec_list, q_len_list, ans_vectors, img_features,a_word_vec,ans_space_score, t_qid_list, img_ids, epoch = dataloader.next_batch(config.BATCH_SIZE)
                #print 'data provide time', time.time()-data_time
                feed_dict = {self.model.q_input: q_word_vec_list, self.model.a_input: a_word_vec,self.model.ans1: ans_vectors,self.model.ans_space_score:ans_space_score,
                             self.model.seqlen: q_len_list, self.model.img_vec: img_features,
                             self.lr: config.VQA_LR, self.model.keep_prob: config.KEEP_PROB, self.model.is_training: True}
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _,vqadecay_lr,global_step = sess.run([vqa_train_op,vqa_decay_lr,self.global_step], feed_dict=feed_dict)
                to_summary, tc_summary, ta_summary, acc, o_loss, c_loss, predict1,af = sess.run(
                    [train_ol_summary, train_ce_summary, train_acc_summary, accuracy, self.overall_loss, cross_entrophy,
                     vqa_predict1,af_distance], options=run_options,
                    run_metadata=run_metadata, feed_dict=feed_dict)
            else:
                q_strs, q_word_vec_list, q_len_list, ans_vectors, img_features,a_word_vec,ans_space_score, q_strs2, q_word_vec_list2, q_len_list2, ans_vectors2, img_features2, a_word_vec2,t_qid_list, img_ids, epoch = dataloader_pairs.next_batch(config.BATCH_SIZE)
                feed_dict = {self.model.q_input: q_word_vec_list, self.model.a_input: a_word_vec, self.model.ans1: ans_vectors,self.model.ans_space_score:ans_space_score,
                             self.model.seqlen: q_len_list, self.model.img_vec: img_features, self.model.a_input2: a_word_vec2,
                             self.lr: config.VQA_LR, self.model.keep_prob: config.KEEP_PROB,self.model.is_training: True,
                             self.model.ans2: ans_vectors2, self.model.img_vec2: img_features2}
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, vqadecay_lr, global_step = sess.run([vqa_train_op, vqa_decay_lr, self.global_step],
                                                       feed_dict=feed_dict)
                to_summary, tc_summary, ta_summary, acc, o_loss, c_loss, predict1, predict2,af,vqad,d_loss,dpn,dapn = sess.run(
                    [train_ol_summary, train_ce_summary, train_acc_summary, accuracy, self.overall_loss, cross_entrophy,
                     vqa_predict1, vqa_predict2,af_distance,vqa_distance,distance_loss,pn_distance,apn_distance], options=run_options,
                    run_metadata=run_metadata, feed_dict=feed_dict)
            train_writer.add_summary(to_summary, i)
            train_writer.add_summary(tc_summary, i)
            train_writer.add_summary(ta_summary, i)
            if (i+1)%config.SHOW_STEP==0 or i ==1:
                cost_t=time.time()-start_t
                print '\n Train Process: Step: %d|%d, Overall Loss: %.7f, Sigmoid Cross Entrophy: %.7f, lr: %.7f, decay_lr: %.7f,accuracy: %.3f, time:%d s, distance1: %.7f, distance2: %.7f, distance_loss: %.7f, pn_distance: %.7f, apn_distance: %.7f' \
                      % (
                          global_step + 1, config.MAX_TRAIN_STEP, o_loss, c_loss, config.VQA_LR, vqadecay_lr, acc,
                          cost_t,
                          af, vqad, d_loss, dpn, dapn)
                print 'Question: %s' % (q_strs[0])
                answer = ans_vectors[0].argmax(axis=0)
                print 'Answer:', dataloader.vec_to_answer(answer), answer
                v_pred1 = predict1[0]
                print 'predict:', dataloader.vec_to_answer(v_pred1), v_pred1
                answer2 = ans_vectors2[0].argmax(axis=0)
                print 'Answer2:', dataloader.vec_to_answer(answer2), answer2
                v_pred2 = predict2[0]
                print 'predict2:', dataloader.vec_to_answer(v_pred2), v_pred2
                start_t = time.time()
                fshow = file('show.txt', 'a+')
                fshow.write(
                    '\n Train Process: Step: %d|%d, Overall Loss: %.7f, Sigmoid Cross Entrophy: %.7f, lr: %.7f, decay_lr: %.7f,accuracy: %.3f, time:%d s, distance1: %.7f, distance2: %.7f, distance_loss: %.7f, pn_distance: %.7f, apn_distance: %.7f' \
                      % (
                          global_step + 1, config.MAX_TRAIN_STEP, o_loss, c_loss, config.VQA_LR, vqadecay_lr, acc,
                          cost_t,
                          af, vqad, d_loss, dpn, dapn))
                fshow.close()
            if (i+1)%config.VAL_STEP==0:
                val_ave_l,val_ave_acc,acc_per_question_type,acc_per_answer_type=self.exec_validation(sess,mode='val',it=i,folder=config.FOLDER_NAME)
                vl_summary, va_summary = sess.run([val_average_l_summary, val_average_acc_summary],
                                                  feed_dict={val_average_loss: val_ave_l,
                                                             val_average_accuracy: val_ave_acc})
                print 'the average validation losses is %.7f, average accuracy is %.5f' % \
                      (val_ave_l, val_ave_acc)
                print 'per answer:',acc_per_answer_type
                print 'per quetion type:',acc_per_question_type
                valid_writer.add_summary(vl_summary, i)
                valid_writer.add_summary(va_summary, i)
                if val_ave_acc>best_result[0]:
                    print 'the previous max val acc is', best_result[0], 'at iter',best_result[1]
                    best_result[0]=val_ave_acc
                    best_result[1]=i
                    print 'now the best result is',best_result[0]
                else:
                    print 'the best result is',best_result[0],'at iter',best_result[1]
                    val_counter+=1
                f = file('result.txt','a+')
                f.write('\n' + 'the step is %d,the average validation losses is %.7f, average accuracy is %.5f' % \
                      (i+1,val_ave_l, val_ave_acc)+'\n'+'per quetion type: %s'%(acc_per_answer_type))
                f.close()
            if val_counter>5:
                self.save_vqa_model(sess,config.VQA_SCOPE,i+1)
                break
            if (i+1)%config.SAVE_STEP==0:
                print 'saving model .......'
                self.save_vqa_model(sess,config.VQA_SCOPE,i+1) # Save VQA Model

    def exec_validation(self,sess,mode,folder, it=0, visualize=False):

        dp = VQADataLoader(mode=mode, batchsize=config.VAL_BATCH_SIZE, folder=folder)
        total_questions = len(dp.getQuesIds())
        epoch = 0
        pred_list = []
        testloss_list = []
        stat_list = []
        while epoch == 0:
            q_strs, q_word_vec_list, q_len_list, ans_vectors, img_features,a_word_vec,ans_score,ans_space_score, t_qid_list, img_ids, epoch = dp.next_batch(config.BATCH_SIZE)
            feed_dict = {self.model.q_input: q_word_vec_list, self.model.ans1: ans_vectors,
                         self.model.seqlen: q_len_list, self.model.img_vec: img_features,
                         self.lr: config.VQA_LR, self.model.keep_prob: 1.0, self.model.is_training: False}

            t_predict_list,predict_loss=sess.run([self.model.predict1, self.model.softmax_cross_entrophy1], feed_dict=feed_dict)
            t_pred_str = [dp.vec_to_answer(pred_symbol) for pred_symbol in t_predict_list]
            testloss_list.append(predict_loss)
            ans_vectors=np.asarray(ans_vectors).argmax(1)
            for qid, iid, ans, pred in zip(t_qid_list, img_ids, ans_vectors, t_pred_str):
                # pred_list.append({u'answer':pred, u'question_id': int(dp.getStrippedQuesId(qid))})
                pred_list.append((pred, int(dp.getStrippedQuesId(qid))))
                if visualize:
                    q_list = dp.seq_to_list(dp.getQuesStr(qid))
                    if mode == 'test-dev' or 'test':
                        ans_str = ''
                        ans_list = [''] * 10
                    else:
                        ans_str = dp.vec_to_answer(ans)
                        ans_list = [dp.getAnsObj(qid)[i]['answer'] for i in xrange(10)]
                    stat_list.append({ \
                        'qid': qid,
                        'q_list': q_list,
                        'iid': iid,
                        'answer': ans_str,
                        'ans_list': ans_list,
                        'pred': pred})
            percent = 100 * float(len(pred_list)) / total_questions
            sys.stdout.write('\r' + ('%.2f' % percent) + '%')
            sys.stdout.flush()

        print 'Deduping arr of len', len(pred_list)
        deduped = []
        seen = set()
        for ans, qid in pred_list:
            if qid not in seen:
                seen.add(qid)
                deduped.append((ans, qid))
        print 'New len', len(deduped)
        final_list = []
        for ans, qid in deduped:
            final_list.append({u'answer': ans, u'question_id': qid})

        mean_testloss = np.array(testloss_list).mean()

        if mode == 'val':
            valFile = './%s/val2015_resfile_%d' % (folder,it)
            with open(valFile, 'w') as f:
                json.dump(final_list, f)
            if visualize:
                visualize_failures(stat_list, mode)
            annFile = config.DATA_PATHS['val']['ans_file']
            quesFile = config.DATA_PATHS['val']['ques_file']
            vqa = VQA(annFile, quesFile)
            vqaRes = vqa.loadRes(valFile, quesFile)
            vqaEval = VQAEval(vqa, vqaRes, n=2)
            vqaEval.evaluate()
            acc_overall = vqaEval.accuracy['overall']
            acc_perQuestionType = vqaEval.accuracy['perQuestionType']
            acc_perAnswerType = vqaEval.accuracy['perAnswerType']
            return mean_testloss, acc_overall, acc_perQuestionType, acc_perAnswerType
        elif mode == 'test-dev':
            filename = './%s/vqa_OpenEnded_mscoco_test-dev2015_%s-%d-' % (folder, folder,it) + str(it).zfill(8) + '_results'
            with open(filename + '.json', 'w') as f:
                json.dump(final_list, f)
            if visualize:
                visualize_failures(stat_list, mode)
        elif mode == 'test':
            filename = './%s/vqa_OpenEnded_mscoco_test2015_%s-%d-' % (folder, folder,it) + str(it).zfill(8) + '_results'
            with open(filename + '.json', 'w') as f:
                json.dump(final_list, f)
            if visualize:
                visualize_failures(stat_list, mode)

    def eval(self):

        #################### Session Setting#########################
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        self.load_vqa_model(sess, config.VQA_SCOPE)
        self.exec_validation(sess, mode='test-dev', folder=config.FOLDER_NAME)

    def train_eval(self):
        #################### Basic Operations#########################
        if config.POSITIVE_ONLY:
            self.overall_loss = self.model.loss1
            cross_entrophy = self.model.sigmoid_cross_entrophy1
            accuracy = self.model.accuracy1
            vqa_predict1 = self.model.predict1
            af_distance = self.model.distance
        else:
            self.overall_loss = self.model.overloss
            cross_entrophy = self.model.sigmoid_cross_entrophy1 + self.model.sigmoid_cross_entrophy2
            accuracy = (self.model.accuracy1 + self.model.accuracy2) / 2
            vqa_predict1 = self.model.predict1
            vqa_predict2 = self.model.predict2
            af_distance = self.model.distance
            vqa_distance = self.model.distance2
            distance_loss = self.model.distance_loss
            pn_distance = self.model.dpn
            apn_distance = self.model.dapn

        vqa_train_op, vqa_decay_lr = self.build_train_ops()
        #################### Summary Setting #########################
        val_average_loss = tf.placeholder('float')
        val_average_accuracy = tf.placeholder('float')

        train_ol_summary = tf.summary.scalar('train_overall_loss', self.overall_loss)
        train_ce_summary = tf.summary.scalar('train_cross_entrophy', cross_entrophy)
        train_acc_summary = tf.summary.scalar('accuracy', accuracy)

        val_average_l_summary = tf.summary.scalar('val_ave_cross_entrophy', val_average_loss)
        val_average_acc_summary = tf.summary.scalar('val_average_accuracy', val_average_accuracy)

        #################### Session Setting#########################
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(config.log_path + 'train', sess.graph)
        valid_writer = tf.summary.FileWriter(config.log_path + 'valid')

        if config.USE_PRE_VQA_MODEL:
            self.load_vqa_model(sess, config.VQA_SCOPE)
        #################### Train Step Function#########################

        dataloader = VQADataLoader(batchsize=config.BATCH_SIZE, mode='train')
        dataloader_pairs = VQADataLoader_pair(batchsize=config.BATCH_SIZE, mode='train')
        start_t = time.time()
        best_result = [0.0, 0]
        val_counter = 0
        for i in range(config.MAX_TRAIN_STEP):
            if config.POSITIVE_ONLY:
                # data_time=time.time()
                q_strs, q_word_vec_list, q_len_list, ans_vectors, img_features, a_word_vec, ans_space_score, t_qid_list, img_ids, epoch = dataloader.next_batch(
                    config.BATCH_SIZE)
                # print 'data provide time', time.time()-data_time
                feed_dict = {self.model.q_input: q_word_vec_list, self.model.a_input: a_word_vec,
                             self.model.ans1: ans_vectors, self.model.ans_space_score: ans_space_score,
                             self.model.seqlen: q_len_list, self.model.img_vec: img_features,
                             self.lr: config.VQA_LR, self.model.keep_prob: config.KEEP_PROB,
                             self.model.is_training: True}
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, vqadecay_lr, global_step = sess.run([vqa_train_op, vqa_decay_lr, self.global_step],
                                                       feed_dict=feed_dict)
                to_summary, tc_summary, ta_summary, acc, o_loss, c_loss, predict1, af = sess.run(
                    [train_ol_summary, train_ce_summary, train_acc_summary, accuracy, self.overall_loss, cross_entrophy,
                     vqa_predict1, af_distance], options=run_options,
                    run_metadata=run_metadata, feed_dict=feed_dict)
            else:
                q_strs, q_word_vec_list, q_len_list, ans_vectors, img_features, a_word_vec, ans_space_score, q_strs2, q_word_vec_list2, q_len_list2, ans_vectors2, img_features2, a_word_vec2, t_qid_list, img_ids, epoch = dataloader_pairs.next_batch(
                    config.BATCH_SIZE)
                feed_dict = {self.model.q_input: q_word_vec_list, self.model.a_input: a_word_vec,
                             self.model.ans1: ans_vectors, self.model.ans_space_score: ans_space_score,
                             self.model.seqlen: q_len_list, self.model.img_vec: img_features,
                             self.model.a_input2: a_word_vec2,
                             self.lr: config.VQA_LR, self.model.keep_prob: config.KEEP_PROB,
                             self.model.is_training: True,
                             self.model.ans2: ans_vectors2, self.model.img_vec2: img_features2}
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, vqadecay_lr, global_step = sess.run([vqa_train_op, vqa_decay_lr, self.global_step],
                                                       feed_dict=feed_dict)
                to_summary, tc_summary, ta_summary, acc, o_loss, c_loss, predict1, predict2, af, vqad, d_loss, dpn, dapn = sess.run(
                    [train_ol_summary, train_ce_summary, train_acc_summary, accuracy, self.overall_loss, cross_entrophy,
                     vqa_predict1, vqa_predict2, af_distance, vqa_distance, distance_loss, pn_distance, apn_distance],
                    options=run_options,
                    run_metadata=run_metadata, feed_dict=feed_dict)
            train_writer.add_summary(to_summary, i)
            train_writer.add_summary(tc_summary, i)
            train_writer.add_summary(ta_summary, i)
            if (i + 1) % config.SHOW_STEP == 0 or i == 1:
                cost_t = time.time() - start_t
                print '\n Train Process: Step: %d|%d, Overall Loss: %.7f, Sigmoid Cross Entrophy: %.7f, lr: %.7f, decay_lr: %.7f,accuracy: %.3f, time:%d s, distance1: %.7f, distance2: %.7f, distance_loss: %.7f, pn_distance: %.7f, apn_distance: %.7f' \
                      % (
                          global_step + 1, config.MAX_TRAIN_STEP, o_loss, c_loss, config.VQA_LR, vqadecay_lr, acc,
                          cost_t,
                          af, vqad, d_loss, dpn, dapn)
                print 'Question: %s' % (q_strs[0])
                answer = ans_vectors[0].argmax(axis=0)
                print 'Answer:', dataloader.vec_to_answer(answer), answer
                v_pred1 = predict1[0]
                print 'predict:', dataloader.vec_to_answer(v_pred1), v_pred1
                answer2 = ans_vectors2[0].argmax(axis=0)
                print 'Answer2:', dataloader.vec_to_answer(answer2), answer2
                v_pred2 = predict2[0]
                print 'predict2:', dataloader.vec_to_answer(v_pred2), v_pred2
                start_t = time.time()
                fshow = file('show.txt', 'a+')
                fshow.write(
                    '\n Train Process: Step: %d|%d, Overall Loss: %.7f, Sigmoid Cross Entrophy: %.7f, lr: %.7f, decay_lr: %.7f,accuracy: %.3f, time:%d s, distance1: %.7f, distance2: %.7f, distance_loss: %.7f, pn_distance: %.7f, apn_distance: %.7f' \
                    % (
                        global_step + 1, config.MAX_TRAIN_STEP, o_loss, c_loss, config.VQA_LR, vqadecay_lr, acc,
                        cost_t,
                        af, vqad, d_loss, dpn, dapn))
                fshow.close()
            if (i + 1) % config.VAL_STEP == 0:
               self.exec_validation(sess, mode='test',it=i,folder=config.FOLDER_NAME)

            if (i+1)%config.SAVE_STEP==0:
                print 'saving model .......'
                self.save_vqa_model(sess,config.VQA_SCOPE,i+1) # Save VQA Model









