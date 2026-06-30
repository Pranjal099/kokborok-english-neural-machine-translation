import sentencepiece as spm

# Load tokenizer model
sp = spm.SentencePieceProcessor()
sp.load('kok_eng_tokenizer.model')  # Ensure this file exists in your folder

def tokenize_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            token_ids = sp.encode(line.strip(), out_type=int)
            fout.write(' '.join(map(str, token_ids)) + '\n')

# Tokenize your files
tokenize_file('train.kok', 'train.tok.kok')
tokenize_file('train.eng', 'train.tok.eng')
tokenize_file('val.kok', 'val.tok.kok')
tokenize_file('val.eng', 'val.tok.eng')
tokenize_file('test.kok', 'test.tok.kok')
tokenize_file('test.eng', 'test.tok.eng')
