import transformers
from transformers import BertTokenizer, BertModel

DEVICE = "cuda"
MAX_LEN = 64
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 5
BERT_PATH = BertModel.from_pretrained("bert-base-multilingual-cased")
MODEL_PATH = "model.bin"
TOKENIZER = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

TRAIN_PROC = ""
DEVEL_PROC = ""
EVAL_PROC = ""

