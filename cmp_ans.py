import config
import json, random
QID_KEY_SEPARATOR = '/'

data_split = 'train'
qdic, adic, qidpairs = {}, {}, {}

a = open(config.DATA_PATHS[data_split]['question_pair'], 'r')
pairs = json.load(a)
for i in pairs:
    qidpairs[data_split + QID_KEY_SEPARATOR + str(i[0])] = data_split + QID_KEY_SEPARATOR + str(i[1])
    qidpairs[data_split + QID_KEY_SEPARATOR + str(i[1])] = data_split + QID_KEY_SEPARATOR + str(i[0])

b = open(config.DATA_PATHS[data_split]['ques_file'], 'r')
qdata = json.load(b)['questions']
for q in qdata:
    if qidpairs.has_key(data_split + QID_KEY_SEPARATOR + str(q['question_id'])):
        qdic[data_split + QID_KEY_SEPARATOR + str(q['question_id'])] = \
            {'qstr': q['question'], 'iid': q['image_id']}

c = open(config.DATA_PATHS[data_split]['ans_file'], 'r')
adata = json.load(c)['annotations']
for a in adata:
    if qidpairs.has_key(data_split + QID_KEY_SEPARATOR + str(a['question_id'])):
        adic[data_split + QID_KEY_SEPARATOR + str(a['question_id'])] = \
            a['answers']

print 'parsed', len(qdic), 'questions for', data_split


def getQuesIds():
    return qdic.keys()
def getqid2(qid):
    return qidpairs[qid]
def getans(qid):
    ans = []
    for i in range(10):
        ans.append(adic[qid][i]['answer'])
    return ans
def getQuesStr( qid):
    return qdic[qid]['qstr']

qid_list = getQuesIds()
random.shuffle(qid_list)
count = 0
for j,qid in enumerate(qid_list):
    ques = getQuesStr(qid)
    ans1 = getans(qid)
    qid2 = getqid2(qid)
    ans2 = getans(qid2)
    print 'ques:  ',ques
    print 'ans1:',ans1
    print 'ans2:', ans2
    if j == 99:
        break