import importlib.metadata
import re
import importlib
#fast BPE TOKENISER
import tiktoken
print("Tiktoken version: ", importlib.metadata.version("tiktoken"))

#instantiating BPE tokeniser from tiktoken, the use of it is similar to simpletokeniserV2 encode, decode method.
bytePairTokeniser = tiktoken.get_encoding("gpt2")

text = ("hello, do you like tea? <|endoftext|> In the sunlit terraces"
       "of some unknown place.")
#encode
integers = bytePairTokeniser.encode(text, allowed_special = {"<|endoftext|>"}) 
print(integers)

#decode
strings = bytePairTokeniser.decode(integers)
print(strings)

#creating input, target pair -> we will implement a data loader the fetches input, target pair using a sliding window approach.

with open("/Users/rudxkush/LLM/the-verdict.txt", "r", encoding="utf-8") as f:  #opening the file in the read mode and storing it in the raw_text variable. 
    raw_text = f.read()
enc_text = bytePairTokeniser.encode(raw_text)
print(len(enc_text)) #no of tokens in the training set.

enc_sample = enc_text[50:] #to  keep things interesting we are removing the first 50 tokens.

#Intuitive approach
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")

#processing inputs along with the targets, which are the input shifted by 1 position.
for i in range(1, context_size+1):
    context = enc_sample[0:i] # till i so for iteration 1 it is just till 1 that means only one element going to show up that is enc_sample[0]
    desired = enc_sample[i]

    print(context, "------>", desired)

for i in range(1, context_size+1):
    context = enc_sample[0:i] 
    desired = enc_sample[i]

    print(bytePairTokeniser.decode(context), "---->", bytePairTokeniser.decode([desired]))


