__author__ = 'antony'
from  Solver import Solver
from Model import Baseline
import tensorflow as tf
import config,os
from new_data_loader import VQADataLoader
from pairwaise_dataloader import VQADataLoader_pair
import json
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=config.GPU_ID


def make_answer_vocab(adic, vocab_size):
    """
    Returns a dictionary that maps words to indices.
    """
    adict = {'': 0}
    nadict = {'': 1000000}
    vid = 1
    for qid in adic.keys():
        answer_obj = adic[qid]
        answer_list = [ans['answer'] for ans in answer_obj]

        for q_ans in answer_list:
            # create dict
            if adict.has_key(q_ans):
                nadict[q_ans] += 1
            else:
                nadict[q_ans] = 1
                adict[q_ans] = vid
                vid += 1

    # debug
    nalist = []
    for k, v in sorted(nadict.items(), key=lambda x: x[1]):
        nalist.append((k, v))

    # remove words that appear less than once
    n_del_ans = 0
    n_valid_ans = 0
    adict_nid = {}
    for i, w in enumerate(nalist[:-vocab_size]):
        del adict[w[0]]
        n_del_ans += w[1]
    for i, w in enumerate(nalist[-vocab_size:]):
        n_valid_ans += w[1]
        adict_nid[w[0]] = i

    return adict_nid


def make_question_vocab(qdic):
    """
    Returns a dictionary that maps words to indices.
    """
    vdict = {'': 0}
    vid = 1
    for qid in qdic.keys():
        # sequence to list
        q_str = qdic[qid]['qstr']
        q_list = VQADataLoader.seq_to_list(q_str)

        # create dict
        for w in q_list:
            if not vdict.has_key(w):
                vdict[w] = vid
                vid += 1
    return vdict

def make_vocab_files():
    """
    Produce the question and answer vocabulary files.
    """
    print 'making question vocab...', config.QUESTION_VOCAB_SPACE
    qdic, _= VQADataLoader.load_data(config.QUESTION_VOCAB_SPACE)
    question_vocab = make_question_vocab(qdic)
    print 'making answer vocab...', config.ANSWER_VOCAB_SPACE
    _, adic = VQADataLoader.load_data(config.ANSWER_VOCAB_SPACE)
    answer_vocab = make_answer_vocab(adic, config.NUM_OUTPUT_UNITS)
    return question_vocab, answer_vocab

def main():
    folder = config.FOLDER_NAME
    if not os.path.exists('./%s' % folder):
        os.makedirs('./%s' % folder)
    question_vocab, answer_vocab = {}, {}
    if os.path.exists('./%s/vdict.json' % folder) and os.path.exists('./%s/adict.json' % folder):
        print 'restoring vocab'
        with open('./%s/vdict.json' % folder, 'r') as f:
            question_vocab = json.load(f)
        with open('./%s/adict.json' % folder, 'r') as f:
            answer_vocab = json.load(f)
    else:
        question_vocab, answer_vocab = make_vocab_files()
        with open('./%s/vdict.json' % folder, 'w') as f:
            json.dump(question_vocab, f)
        with open('./%s/adict.json' % folder, 'w') as f:
            json.dump(answer_vocab, f)

    print 'question vocab size:', len(question_vocab)
    print 'answer vocab size:', len(answer_vocab)
    with tf.variable_scope(config.VQA_SCOPE):
        model=Baseline()
    solver=Solver(model)
    solver.train()


if __name__ == '__main__':
	main() 
