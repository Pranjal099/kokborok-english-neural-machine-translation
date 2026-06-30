import sentencepiece as spm

# Prepare training data by combining Kokborok and English into one file
with open('combined_train.txt', 'w', encoding='utf-8') as outfile:
    for fname in ['train.kok', 'train.eng']:
        with open(fname, 'r', encoding='utf-8') as infile:
            for line in infile:
                outfile.write(line)

# Train SentencePiece tokenizer
spm.SentencePieceTrainer.Train(
    input='combined_train.txt',
    model_prefix='kok_eng_tokenizer',
    vocab_size=32000,
    character_coverage=1.0,
    model_type='bpe'  # or 'unigram', 'word', 'char'
)
