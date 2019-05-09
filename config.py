ROOT='/home/zhouyiyi/VQA/DATA/'
SAVE_ROOT=ROOT+'CAPS_VQA/VQA2/baseline_v1_s2_v2/'
DATA_ROOT='/home/zhouyiyi/VQA/DATA/preprocessed_files/coco/'

# what data to use for training
TRAIN_DATA_SPLITS = 'train+val'
# what data to use for the vocabulary
QUESTION_VOCAB_SPACE = 'train+val'
ANSWER_VOCAB_SPACE = 'train+val' # test/test-dev/genome should not appear here
PREFIX={
    'vg':'/home/zhouyiyi/VQA/DATA/image_features/visualgenome/resnet_res5c_bgrms_large/VG_100K_2/',
    'vqa-train':'/home/zhouyiyi/VQA/DATA/image_features/vqa_test_res5c/'+'resnet_res5c_bgrms_large/train2014/COCO_train2014_',
    'vqa-val':'/home/zhouyiyi/VQA/DATA/image_features/vqa_test_res5c/'+'resnet_res5c_bgrms_large/val2014/COCO_val2014_',
    'vqa-test':'/home/zhouyiyi/VQA/DATA/image_features/vqa_test_res5c/'+'resnet_res5c_bgrms_large/test2015/COCO_test2015_',
    'coco_train':'/home/zhouyiyi/VQA/DATA/image_features/vqa_test_res5c/'+'resnet_res5c_bgrms_large/train2014/COCO_train2014_',
    'coco_test':'/home/zhouyiyi/VQA/DATA/image_features/vqa_test_res5c/'+'resnet_res5c_bgrms_large/val2014/COCO_val2014_'
}
feat = 'res5c'

# vqa tools - get from https://github.com/VT-vision-lab/VQA
VQA_TOOLS_PATH = '/home/zhouyiyi/VQA/CODE/PythonHelperTools'
VQA_EVAL_TOOLS_PATH = '/home/zhouyiyi/VQA/CODE/PythonEvaluationTools'
ANS_GLOVE = "/home/lxm/Data/ans_glove3000.json"

# location of the data
VQA_PREFIX = '/home/zhouyiyi/VQA/DATA/preprocessed_files/new'
DATA_PREFIX = '/home/zhouyiyi/VQA/DATA'
PAIRS_PREFIX = '/home/lxm/Data'
DATA_PATHS = {
	'train': {
		'ques_file': VQA_PREFIX + '/v2_OpenEnded_mscoco_train2014_questions.json',
		'ans_file': VQA_PREFIX + '/v2_mscoco_train2014_annotations.json',
		'features_prefix': DATA_PREFIX + '/image_features/vqa_test_res5c/'+'resnet_res5c_bgrms_large/train2014/COCO_train2014_',
		'question_pair': PAIRS_PREFIX  + '/v2_mscoco_train2014_complementary_pairs.json'
	},
	'val': {
		'ques_file': VQA_PREFIX + '/v2_OpenEnded_mscoco_val2014_questions.json',
		'ans_file': VQA_PREFIX + '/v2_mscoco_val2014_annotations.json',
		'features_prefix': DATA_PREFIX + '/image_features/vqa_test_res5c/'+'resnet_res5c_bgrms_large/val2014/COCO_val2014_',
		'question_pair': PAIRS_PREFIX  + '/v2_mscoco_val2014_complementary_pairs.json'
	},
	'test-dev': {
		'ques_file': VQA_PREFIX + '/v2_OpenEnded_mscoco_test-dev2015_questions.json',
		'features_prefix': DATA_PREFIX + '/image_features/vqa_test_res5c/'+'resnet_res5c_bgrms_large/test2015/COCO_test2015_'
	},
	'test': {
		'ques_file': VQA_PREFIX + '/v2_OpenEnded_mscoco_test2015_questions.json',
		'features_prefix': DATA_PREFIX + '/image_features/vqa_test_res5c/'+'resnet_res5c_bgrms_large/test2015/COCO_test2015_'
	},
	'genome': {
		'genome_file': VQA_PREFIX + '/Questions/OpenEnded_genome_train_questions.json',
		'features_prefix': DATA_PREFIX + '/image_features/visualgenome/resnet_res5c_bgrms_large/VG_100K_2/'
	}
}
VAL_NUM=5000

# VQA Setting
CNN_DIM=2048
GLOVE_DIM = 300
VQA_ANS_OUTPUT = 3000      #the top 3000 answers
NUM_OUTPUT_UNITS=3000
QUESTION_MAX_LENGTH = 15
QUESTION_RNN_DIM = 2048
FF_DIM=2048
IMAGE_SCALE = 14
CNN_REGION_NUM = 196
KEEP_PROB = 0.8
ATTENT_DIM = 1024
USE_ATTENTION_DROP = False
COMBINED_MLP_DROP = True
DECAY_RATE= 0.5
DECAY_STEPS = 100
MAX_EPOCHS = 50
MAX_STEPS = 10000
WEIGHT_DECAY = 0.00005
LEARNING_RATE = 0.0005

MFB_DIM=5000
MFB_OUT_DIM=1000
MFB_FACTOR_NUM=5

MODEL_SAVE_NAME='Coco_BV1'
MODEL_LOAD_NAME='Coco_BV1-70000'
VQA_SCOPE='vqa'
INITIAL_LEARNING_RATE=0.0001
VQA_LR=0.0001
LEARNING_RATE_DECAY_FACTOR=0.5
CLIP_GRADIENT=10.0
MODEL_NAME='Coco_BV1'

#Trainging
TRAIN_THRESHOLD=2.0
VAL_STEP=5000
MODEL_SAVE_ROOT=SAVE_ROOT+'models/'
SUMMARY_DIR=SAVE_ROOT+'logs/'
MAX_TRAIN_STEP=20000
NUM_STEPS_PER_DECAY=5000
SAVE_STEP=5000
BATCH_SIZE=32
VAL_BATCH_SIZE=32
SHOW_STEP=100
USE_PRE_VQA_MODEL=True
VALIDATE=True
POSITIVE_ONLY=False

FOLDER_NAME='baseline_%s'%TRAIN_DATA_SPLITS
log_path = FOLDER_NAME + '/logs/'
EVALUATION_SAVE_PATH=SUMMARY_DIR+'%s_results.json'%MODEL_NAME

GPU_ID='3'

"""
NETOWKR SETTING
"""
fc_kernel_initializer_scale = 0.08
fc_kernel_regularizer_scale = 1e-4
fc_activity_regularizer_scale = 0.0
conv_kernel_regularizer_scale = 1e-4
conv_activity_regularizer_scale = 0.0
fc_drop_rate = 0.5