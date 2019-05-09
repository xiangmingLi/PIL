__author__ = 'antony'
import config
import json,re,os
import numpy as np
import string
from PIL import Image
from PIL import ImageFont, ImageDraw
import shutil

QID_KEY_SEPARATOR='/'

"""
Load Data
"""


def visualize_failures(stat_list, mode):
    def save_qtype(qtype_list, save_filename, mode):

        if mode == 'val':
            savepath = os.path.join('./eval', save_filename)
            # TODO
            img_pre = '/home/dhpseth/vqa/02_tools/VQA/Images/val2014'
        elif mode == 'test-dev':
            savepath = os.path.join('./test-dev', save_filename)
            # TODO
            img_pre = '/home/dhpseth/vqa/02_tools/VQA/Images/test2015'
        elif mode == 'test':
            savepath = os.path.join('./test', save_filename)
            # TODO
            img_pre = '/home/dhpseth/vqa/02_tools/VQA/Images/test2015'
        else:
            raise Exception('Unsupported mode')
        if os.path.exists(savepath): shutil.rmtree(savepath)
        if not os.path.exists(savepath): os.makedirs(savepath)

        for qt in qtype_list:
            count = 0
            for t_question in stat_list:
                # print count, t_question
                if count < 40 / len(qtype_list):
                    t_question_list = t_question['q_list']
                    saveflag = False
                    # print 'debug****************************'
                    # print qt
                    # print t_question_list
                    # print t_question_list[0] == qt[0]
                    # print t_question_list[1] == qt[1]
                    if t_question_list[0] == qt[0] and t_question_list[1] == qt[1]:
                        saveflag = True
                    else:
                        saveflag = False

                    if saveflag == True:
                        t_iid = t_question['iid']
                        if mode == 'val':
                            t_img = Image.open(os.path.join(img_pre, \
                                                            'COCO_val2014_' + str(t_iid).zfill(12) + '.jpg'))
                        elif mode == 'test-dev' or 'test':
                            t_img = Image.open(os.path.join(img_pre, \
                                                            'COCO_test2015_' + str(t_iid).zfill(12) + '.jpg'))

                        # for caption
                        # print t_iid
                        # annIds = caps.getAnnIds(t_iid)
                        # anns = caps.loadAnns(annIds)
                        # cap_list = [ann['caption'] for ann in anns]
                        ans_list = t_question['ans_list']
                        draw = ImageDraw.Draw(t_img)
                        for i in range(len(ans_list)):
                            try:
                                draw.text((10, 10 * i), str(ans_list[i]))
                            except:
                                pass

                        ans = t_question['answer']
                        pred = t_question['pred']
                        if ans == -1:
                            pre = ''
                        elif ans == pred:
                            pre = 'correct  '
                        else:
                            pre = 'failure  '
                        # print ' aaa ', ans, pred
                        ans = re.sub('/', ' ', str(ans))
                        pred = re.sub('/', ' ', str(pred))
                        img_title = pre + str(' '.join(t_question_list)) + '.  a_' + \
                                    str(ans) + ' p_' + str(pred) + '.png'
                        count += 1
                        print os.path.join(savepath, img_title)
                        t_img.save(os.path.join(savepath, img_title))

    print 'saving whatis'
    qt_color_list = [['what', 'color']]
    save_qtype(qt_color_list, 'colors', mode)

    print 'saving whatis'
    qt_whatis_list = [['what', 'is'], ['what', 'kind'], ['what', 'are']]
    save_qtype(qt_whatis_list, 'whatis', mode)

    print 'saving is'
    qt_is_list = [['is', 'the'], ['is', 'this'], ['is', 'there']]
    save_qtype(qt_is_list, 'is', mode)

    print 'saving how many'
    qt_howmany_list = [['how', 'many']]
    save_qtype(qt_howmany_list, 'howmany', mode)
def load_genome_data():
    pass

def load_vqa_data(data_split):

    qdic,adic={},{}

    q_index='oe_ques_file' if config.EXP_MODE=='OpenEnded' else 'mc_ques_file'
    with open(config.DATA_PATHS[data_split][q_index],'r') as f:
        qdata=json.load(f)['questions']
        for q in qdata:
            qdic[data_split+QID_KEY_SEPARATOR+str(q['question_id'])]=\
                {'qstr':q['question'],'iid':q['image_id']}

        if 'test' not in data_split:
            with open(config.DATA_PATHS[data_split]['ans_file'],'r') as f:
                adata=json.load(f)['annotations']
                for a in adata:
                    adic[data_split+QID_KEY_SEPARATOR+str(a['question_id'])]=\
                    a['answers']
    print 'parsed', len(qdic),'question for',data_split
    return qdic,adic


def load_data(data_split_str):

    all_qdic,all_adic={},{}
    for data_split in data_split_str.split('+'):
        assert data_split in config.DATA_PATHS.keys(), 'unknown data split'
        if data_split=='genome':
            qdic,adic=load_genome_data()
            all_qdic.update(qdic)
            all_adic.update(adic)
        else:
            qdic,adic=load_vqa_data(data_split)
            all_qdic.update(qdic)
            all_adic.update(adic)
    return all_qdic,all_adic

def load_img_features(mode):
        img_feature_root=config.DATA_PATHS[mode]['features_root']
        if os.path.exists(os.path.join(img_feature_root,mode+'_img_feature.dict')):
            with open(os.path.join(img_feature_root,mode+'_img_feature.dict'),'r') as f:
                img_dict=json.load(f)
                return img_dict
        pattern=r'.*.jpg.npz'
        files=os.listdir(img_feature_root)
        img_dict={}
        counter=0
        for name in files:
            print counter
            if re.match(pattern,name):
                path=os.path.join(img_feature_root,name)
                img_dict[path]=np.load(path)['x']
                counter+=1
        print '%d image res features load.'%counter
        with open(os.path.join(img_feature_root,mode+'_img_feature.dict'),'w') as f:
            json.dump(img_dict,f)
        return img_dict


"""
Make Vocabulary
"""
def remove_puntuations(s):
    s=s.lower()
    try:
        s=str(s)
    except:
        s=s.encode('ascii',errors='ignore')
    s=s.translate(string.maketrans(string.punctuation," "*len(string.punctuation))).strip()
    return s
def seq_to_list(s):
    s=s.strip()
    t_str=s.lower()
    t_str=remove_puntuations(t_str)
    for i in [r'\?',r'\!',r'\'',r'\"',r'\$',r'\:',r'\@',r'\(',r'\)',r'\,',r'\.',r'\;']:
        t_str = re.sub( i, '', t_str)
    for i in [r'\-',r'\/']:
        t_str = re.sub( i, ' ', t_str)
    q_list = re.sub(r'\?','',t_str.lower()).split()
    return q_list

def text2int(textnum,numwords):
    if textnum == 'hundred':
        return unicode(str(100))
    if textnum == 'thousand':
        return unicode(str(1000))

    current = result = 0
    for word in textnum.split():
        if word not in numwords:
            return textnum
        scale, increment = numwords[word]
        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0
    return unicode(str(result + current))

def create_text2int_dict():
    numwords = {}
    units = [ "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
              "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
              "sixteen", "seventeen", "eighteen", "nineteen" ]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    scales = ["hundred", "thousand", "million", "billion", "trillion"]
    numbers = units + tens + scales

    numwords["and"] = (1, 0)
    for idx, word in enumerate(units):
        numwords[word] = (1, idx)
    for idx, word in enumerate(tens):
        numwords[word] = (1, idx * 10)
    for idx, word in enumerate(scales):
        numwords[word] = (10 ** (idx * 3 or 2), 0)

    return numwords,numbers

def process_answers(answer):
    ELIMINATE=['on','the','a','in','inside','at','it','is','with','near','behind','front','of','next','to','s']
    NUM_STRING=['one','two','three','four','five','six','seven','eight','night','ten','twenty','thirty','forty','fifty','sixty','hundred']
    words=answer.strip().split()
    numwords,numbers=create_text2int_dict()
    word_list=[]
    for w in words:
        if w in ELIMINATE:
            continue
        if w in numbers:
            w=text2int(w,numwords)
        word_list.append(w)
    return ' '.join(word_list)
def make_question_vocab(qdic):

    vdict={' ':0}
    vid=1
    for qid in qdic.keys():
        q_str=qdic[qid]['qstr']
        q_list=seq_to_list(q_str)

        for w in q_list:
            if not vdict.has_key(w):
                vdict[w]=vid
                vid+=1
    return vdict

def make_answer_vocab(adic,vocab_size):

    adict={'':0}
    nadict={'':1000000}
    vid=1
    for qid in adic.keys():
        answer_obj=adic[qid]
        answer_list=[ans['answer'] for ans in answer_obj]

        for q_ans in answer_list:

            if adict.has_key(q_ans):
                nadict[q_ans]+=1
            else:
                nadict[q_ans]=1
                adict[q_ans]=vid
                vid+=1
    nalist=[]
    for k,v in sorted(nadict.items(),key=lambda  x:x[1]):
        nalist.append((k,v))

    n_del_ans=0
    n_valid_ans=0
    adict_nid={}
    for i,w in enumerate(nalist[:-vocab_size]):
        del adict[w[0]]
        n_del_ans+=w[1]
    for i,w in enumerate(nalist[-vocab_size:]):
        n_valid_ans+=w[1]
        adict_nid[w[0]]=i

    return adict_nid

def make_vocab_file():
    print 'making question vocab...',config.QUESTION_VOCAB_SPACE
    qdic,_=load_data(config.QUESTION_VOCAB_SPACE)
    question_vocabulary=make_question_vocab(qdic)
    print 'making answer vocab...',config.ANSWER_VOCAB_SPACE
    _,adic=load_data(config.ANSWER_VOCAB_SPACE)
    answer_vocabulary=make_answer_vocab(adic,config.NUM_OUTPUT_UNITS)
    return question_vocabulary,answer_vocabulary


def decode_captions(captions, idx_to_word):
    if captions.ndim == 1:
        T = captions.shape[0]
        N = 1
    else:
        N, T = captions.shape

    decoded = []
    for i in range(N):
        words = []
        for t in range(T):
            if captions.ndim == 1:
                try:
                    word = idx_to_word[u'%d'%captions[t]]
                except:
                    word ='unk'
            else:
                try:
                    word = idx_to_word[u'%d'%captions[i, t]]
                except:
                    word='unk'
            if word == '<END>':
                words.append('.')
                break
            if word != '<NULL>' and word !='unk':
                words.append(word)
        decoded.append(' '.join(words))
    return decoded

def decode_answer(answer,answer_2_idx):

    return answer_2_idx[u'%d'%int(answer)]

def display_result(ind,idx_to_word,ans_to_word,f,original_caption=None,generated_caption=None,question=None,predict=None,answer=None,img_id=None,q_type=None):
    if question is not None:
        decoded=decode_captions(question,idx_to_word)[0]
        print 'Question %d:'%ind, decoded
        f.write('Question %d:'%ind+''+decoded+'\n')
    if img_id is not None:
        print 'Img id %d:'%ind,img_id
        f.write('Img id %d:'%ind+str(img_id)+'\n')
    if q_type is not None:
        print 'Question Type %d:'%ind,config.Q_TYPES[q_type]
        f.write('Question Type %d:'%ind+config.Q_TYPES[q_type]+'\n')
    if answer is not None:
        f.write('Answer %d:'%ind+ decode_answer(answer,ans_to_word)+'\n')
        print 'Answer %d:'%ind, decode_answer(answer,ans_to_word)

    if predict is not None:
        f.write('Model Predict %d:'%ind+decode_answer(predict,ans_to_word)+'\n')
        print 'Model Predict %d:'%ind,decode_answer(predict,ans_to_word)
    if original_caption is not None:

        decode=decode_captions(original_caption,idx_to_word)[0]
        f.write('Ground Truth Caption %d:'%ind+decode+'\n')
        print 'Ground Truth Caption %d:'%ind, decode
    if generated_caption is not None:
        decode=decode_captions(generated_caption,idx_to_word)[0]
        f.write('Generated Caption %d:'%ind+decode+'\n')
        print 'Generated Caption %d:'%ind, decode
    f.write('--------------------------------------------------')
