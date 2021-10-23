import torch
import transformers


DEVICE = "cuda"
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 5
BERT_PATH = "bert-base-multilingual-cased"
MODEL_PATH = "model.bin"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True, truncation=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
