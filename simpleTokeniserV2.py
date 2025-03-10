import re
#encoder

with open("/Users/rudxkush/LLM/the-verdict.txt", "r", encoding="utf-8") as f:  #opening the file in the read mode and storing it in the raw_text variable. 
    raw_text = f.read()
    
print("Total number of character:", len(raw_text))  #counting total number of characters in this raw_taxt.  
print(raw_text[0:99])  #prints first 100 characters for illustration purposes.

#tokenization
preProcessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)   #splits any given text based on white spaces, commas and periods.
preProcessed = [item for item in preProcessed if item.strip()]  #to remove whitespace -> reduces the memory and computing requirement and it is not always necessary. 
print(len(preProcessed))

#converting tokens in token ids -> to numerically represent each token in python
all_tokens = sorted(set(preProcessed)) #unique tokens all sorted in alphabetically order
all_tokens.extend(["<|endoftext|>","<|unk|>"]) #To handle unknown words

#dictionary of tokens and token ids
vocab = {token: integer for integer, token in enumerate(all_tokens)}
#first enumeration each the tokens in the preprocessed list and then we will assign an integer for each token 
print(len(vocab.items()))
#enumerate commands in python takes all the words and then it assigns an integer to each word in alphabetical order. 

#printing last 5 entries of vocab([unique token -> tokenid])
for i, item in enumerate(list(vocab.items())[-5:]):  
    print(item)

with open("/Users/rudxkush/LLM/the-verdict.txt", "r", encoding="utf-8") as f:  #opening the file in the read mode and storing it in the raw_text variable. 
    raw_text = f.read()
    
print("Total number of character:", len(raw_text))  #counting total number of characters in this raw_taxt.  
print(raw_text[0:99])  #prints first 100 characters for illustration purposes. 

#Let us put the implementation of both the encoder and decoder in a class.
#To handle cases to have the provision to handle the unknown tokens.
class simpleTokenizerV2: 
    #contructor method
    def __init__(self, vocab):     #vocab : list of [unique token -> tokenid]
        #intialise two dictionaries
        self.str_to_int = vocab  
        self.int_to_str = {i: s for s, i in vocab.items()} #take the string and integer in the vocabulary and then for every integer we mention which token it is.
        pass
    
    #encoder -> creating a list of [unique token -> tokenid] 
    def encode(self, text):
       preProcessed = re.split(r'([,.:;?_!"()\']|--|\s)', text) 
       preProcessed = [item for item in preProcessed if item.strip()] 
       preProcessed = [
           item if item in self.str_to_int
           else "<|unk|>" for item in preProcessed
       ]
       ids = [self.str_to_int[s] for s in preProcessed ] # -> tokens are converted into token ids
       return ids
    
    #decoder -> ~encoder
    def decode(self, ids):
        text = " ". join([self.int_to_str[i] for i in ids]) # -> tokens ids are converted into tokens and then we join them togeter with spaces in between.
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)   # get rid of the extra spaces.  
        return text

#creating an instance of the class by passing vocabulary as an input.
tokenizer = simpleTokenizerV2(vocab) 
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = "<|endoftext|>".join((text1, text2))
print(text) #added eot to differtiate between given different contexts.

print(tokenizer.encode(text))
#decoding...
print(tokenizer.decode(tokenizer.encode(text)))
