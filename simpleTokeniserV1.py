import re
#encoder

with open("/Users/rudxkush/LLM/the-verdict.txt", "r", encoding="utf-8") as f:  #opening the file in the read mode and storing it in the raw_text variable. 
    raw_text = f.read()
    
print("Total number of character:", len(raw_text))  #counting total number of characters in this raw_taxt.  
print(raw_text[0:99])  #prints first 100 characters for illustration purposes.

#tokenization

preProcessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)   #splits any given text based on white spaces, commas and periods.
preProcessed = [item for item in preProcessed if item.strip()]  #to remove whitespace -> reduces the memory and computing requirement and it is not always necessary. 
print(preProcessed)
print(len(preProcessed))

#converting tokens in token ids -> to numerically represent each token in python

all_words = sorted(set(preProcessed)) #unique tokens all sorted in alphabetically order
print(len(all_words))

#dictionary of tokens and token ids

vocab = {token: integer for integer, token in enumerate(all_words)}
#enumerate commands in python takes all the words and then it assigns an integer to each word in alphabetical order. 

#printing first 50 [unique token -> tokenid]
for i, item in enumerate(vocab.items()):  
    print(item)
    if i >= 50:
        break

with open("/Users/rudxkush/LLM/the-verdict.txt", "r", encoding="utf-8") as f:  #opening the file in the read mode and storing it in the raw_text variable. 
    raw_text = f.read()
    
print("Total number of character:", len(raw_text))  #counting total number of characters in this raw_taxt.  
print(raw_text[0:99])  #prints first 100 characters for illustration purposes. 

#Let us put the implementation of both the encoder and decoder in a class.
class simpleTokenizerV1: 
    #contructor method
    def __init__(self, vocab):     #vocab : list of [unique token -> tokenid]
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()} #take the string and integer in the vocabulary and then for every integer we mention which token it is.
        pass
    
    #encoder -> creating a list of [unique token -> tokenid] 
    def encode(self, text):
       preProcessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
       preProcessed = [item for item in preProcessed if item.strip()] 
       ids = [self.str_to_int[s] for s in preProcessed ]
       return ids
    
    #decoder -> ~encoder
    def decode(self, ids):
        text = " ". join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

#creating an instance of the class by passing vocabulary as an input.
tokenizer = simpleTokenizerV1(vocab) 
text = """"It's the last he painted, you know,"
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)  #sentence given from the vocab
print(ids)

print(tokenizer.decode(ids))