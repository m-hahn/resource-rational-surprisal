import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large", cache_dir="/u/scr/mhahn/cache/")


# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id, cache_dir="/u/scr/mhahn/cache/").cuda()
print("Finished loading GPT2")

import sys
for text in sys.stdin:
    if len(text.strip()) > 3:
       batch = [text]
       tensors = [tokenizer.encode(" "+text, return_tensors='pt') for text in batch] # below using bos, so should be no need for adding "<|endoftext|> "+
       maxLength = max([x.size()[1] for x in tensors])
       for i in range(len(tensors)):
          tensors[i] = torch.cat([torch.LongTensor([tokenizer.bos_token_id]).view(1,1), tensors[i], torch.LongTensor([tokenizer.eos_token_id for _ in range(maxLength - tensors[i].size()[1])]).view(1, -1)], dim=1)
       tensors = torch.cat(tensors, dim=0)
       predictions, _ = model(tensors.cuda())
# Transformers v 3:
#       predictions = model(tensors.cuda())
#       predictions = predictions["logits"]
#       print(tensors)
#       print("PREDICTIONS", predictions.size())      
       surprisals = torch.nn.CrossEntropyLoss(reduction='none')(predictions[:,:-1].contiguous().view(-1, 50257), tensors[:,1:].contiguous().view(-1).cuda()).view(len(batch), -1)
       surprisals = surprisals.detach().cpu()
 #      print(surprisals, surprisals.size())
       surprisalsCollected = []
       for batchElem in range(len(batch)):
         words = [[]]
         if batchElem == 0:
           print(tensors[batchElem])
         for q in range(1, maxLength+1):
            word = tokenizer.decode(int(tensors[batchElem][q]))
            if batchElem == 0:
               print(q, int(tensors[batchElem][q]), word, maxLength)
            if word == '<|endoftext|>':
                break
            if word.startswith(" ") or q == 0:
                words.append([])
            words[-1].append((word, float(surprisals[batchElem][q-1])))
         # find where last word starts and separately get the surprisals
         surprisalsPast = sum([sum(x[1] for x in y) for y in words[:-1]])
         surprisalsFirstFutureWord = sum(x[1] for x in words[-1])
         if batchElem == 0:
            print(words, "Total", surprisalsPast+surprisalsFirstFutureWord)
         else:
            assert False




