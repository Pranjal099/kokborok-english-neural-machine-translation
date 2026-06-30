import sentencepiece as spm

# Load tokenizer model
sp = spm.SentencePieceProcessor()
sp.load('kok_eng_tokenizer.model')

# Example encoding
text = "your sentence here"
tokens = sp.encode(text, out_type=str)  # tokens as strings
token_ids = sp.encode(text, out_type=int)  # tokens as ids

print("Tokens:", tokens)
print("Token IDs:", token_ids)
