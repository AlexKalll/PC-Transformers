import os
class Config:
    VOCAB_SIZE = None
    PAD_ID=None
    EOS_ID=None
    MAX_LENGTH = 512
    DATASET_NAME = "pubmed"  # "ptb" for penntreebank "opwb" for OpenWebText
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
    DATA_DIR = os.path.join(BASE_DIR, "Data") 
    TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer", "outputs")  
    BATCH_SIZE = 8
    num_workers = 8