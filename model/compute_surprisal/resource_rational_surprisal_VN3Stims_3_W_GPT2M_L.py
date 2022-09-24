import os
import glob
import sys
import random
from collections import defaultdict
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="english")
#parser.add_argument("--load-from-lm", dest="load_from_lm", type=str, default=964163553) # language model taking noised input # Amortized Prediction Posterior
#parser.add_argument("--load-from-autoencoder", dest="load_from_autoencoder", type=str, default=random.choice([647336050, 516252642, 709961927, 727001672, 712478284, 524811876])) # Amortized Reconstruction Posterior
#parser.add_argument("--load-from-plain-lm", dest="load_from_plain_lm", type=str, default=random.choice([27553360, 935649231])) # plain language model without noise (Prior)
parser.add_argument("--load_from_joint", type=str)


# Unique ID for this model run
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))


# Sequence length
parser.add_argument("--sequence_length", type=int, default=random.choice([20]))

# Parameters of the neural network models
parser.add_argument("--batchSize", type=int, default=random.choice([1]))
parser.add_argument("--NUMBER_OF_REPLICATES", type=int, default=random.choice([12,20]))

## Layer size
parser.add_argument("--word_embedding_size", type=int, default=random.choice([512]))
parser.add_argument("--hidden_dim_lm", type=int, default=random.choice([1024]))
parser.add_argument("--hidden_dim_autoencoder", type=int, default=random.choice([512]))

## Layer number
parser.add_argument("--layer_num", type=int, default=random.choice([2]))

## Regularization
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.05]))
parser.add_argument("--weight_dropout_out", type=float, default=random.choice([0.05]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.01]))

## Learning Rates
parser.add_argument("--learning_rate_memory", type = float, default= random.choice([0.00002, 0.00005, 0.0001, 0.0001, 0.0001]))  # Can also use 0.0001, which leads to total convergence to deterministic solution withtin maximum iterations (March 25, 2021)   #, 0.0001, 0.0002 # 1e-7, 0.000001, 0.000002, 0.000005, 0.000007, 
parser.add_argument("--learning_rate_autoencoder", type = float, default= random.choice([0.001, 0.01, 0.1, 0.1, 0.1, 0.1])) # 0.0001, 
parser.add_argument("--lr_decay", type=float, default=random.choice([1.0]))
parser.add_argument("--reward_multiplier_baseline", type=float, default=0.1)
parser.add_argument("--dual_learning_rate", type=float, default=random.choice([0.01, 0.02, 0.05, 0.1, 0.2, 0.3]))
parser.add_argument("--momentum", type=float, default=random.choice([0.5, 0.7, 0.7, 0.7, 0.7, 0.9])) # Momentum is helpful in facilitating convergence to a low-loss solution (March 25, 2021). It might be even more important for getting fast convergence than a high learning rate
parser.add_argument("--entropy_weight", type=float, default=random.choice([0.0])) # 0.0,  0.005, 0.01, 0.1, 0.4]))



# Control
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--tuning", type=int, default=1) #random.choice([0.00001, 0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.0008, 0.001])) # 0.0,  0.005, 0.01, 0.1, 0.4]))

# Lambda and Delta Parameters
parser.add_argument("--deletion_rate", type=float, default=0.5)
parser.add_argument("--predictability_weight", type=float, default=random.choice([0.0, 0.25, 0.5, 0.75, 1.0]))


TRAIN_LM = False
assert not TRAIN_LM



model = "REAL_REAL"

import math

args=parser.parse_args()

############################

assert args.predictability_weight >= 0
assert args.predictability_weight <= 1
assert args.deletion_rate > 0.0
assert args.deletion_rate < 1.0



#############################

assert args.tuning in [0,1]
assert args.batchSize == 1
print(args.myID)
import sys
STDOUT = sys.stdout
print(sys.argv)

print(args)
print(args, file=sys.stderr)



import corpusIteratorWikiWords



def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x


# Load Vocabulary
char_vocab_path = "vocabularies/"+args.language.lower()+"-wiki-word-vocab-50000.txt"

with open(char_vocab_path, "r") as inFile:
     itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
stoi = dict([(itos[i],i) for i in range(len(itos))])


itos_total = ["<SOS>", "<EOS>", "OOV"] + itos
stoi_total = dict([(itos_total[i],i) for i in range(len(itos_total))])


import random
import torch

print(torch.__version__)



class Autoencoder:
  """ Amortized Reconstruction Posterior """
  def __init__(self):
    # This model describes a standard sequence-to-sequence LSTM model with attention
    self.rnn_encoder = torch.nn.LSTM(2*args.word_embedding_size, int(args.hidden_dim_autoencoder/2.0), args.layer_num, bidirectional=True).cuda()
    self.rnn_decoder = torch.nn.LSTM(2*args.word_embedding_size, args.hidden_dim_autoencoder, args.layer_num).cuda()
    self.output = torch.nn.Linear(args.hidden_dim_autoencoder, len(itos)+3).cuda()
    self.word_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=2*args.word_embedding_size).cuda()
    self.logsoftmax = torch.nn.LogSoftmax(dim=2)
    self.softmax = torch.nn.Softmax(dim=2)
    self.attention_softmax = torch.nn.Softmax(dim=1)
    self.train_loss = torch.nn.NLLLoss(ignore_index=0)
    self.print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
    self.char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)
    self.attention_proj = torch.nn.Linear(args.hidden_dim_autoencoder, args.hidden_dim_autoencoder, bias=False).cuda()
    self.attention_proj.weight.data.fill_(0)
    self.output_mlp = torch.nn.Linear(2*args.hidden_dim_autoencoder, args.hidden_dim_autoencoder).cuda()
    self.relu = torch.nn.ReLU()
    self.modules_autoencoder = [self.rnn_decoder, self.rnn_encoder, self.output, self.word_embeddings, self.attention_proj, self.output_mlp]


  def forward(self, input_tensor_pure, input_tensor_noised, target_tensor_onlyNoised, NUMBER_OF_REPLICATES):
      # INPUTS: input_tensor_pure, input_tensor_noised
      # OUTPUT: autoencoder_lossTensor

      autoencoder_embedded = self.word_embeddings(input_tensor_pure[:-1])
      autoencoder_embedded_noised = self.word_embeddings(input_tensor_noised[:-1])
      autoencoder_out_encoder, _ = self.rnn_encoder(autoencoder_embedded_noised, None)
      autoencoder_out_decoder, _ = self.rnn_decoder(autoencoder_embedded, None)
      assert autoencoder_embedded.size()[0] == args.sequence_length-1, (autoencoder_embedded.size()[0], args.sequence_length-1) # Note that this is different from autoencoder2_mlp_bidir_Erasure_SelectiveLoss.py. Would be good if they were unified.
      assert autoencoder_embedded_noised.size()[0] == args.sequence_length-1, (autoencoder_embedded.size()[0], args.sequence_length-1) # Note that this is different from autoencoder2_mlp_bidir_Erasure_SelectiveLoss.py.

      autoencoder_attention = torch.bmm(self.attention_proj(autoencoder_out_encoder).transpose(0,1), autoencoder_out_decoder.transpose(0,1).transpose(1,2))
      autoencoder_attention = self.attention_softmax(autoencoder_attention).transpose(0,1)
      autoencoder_from_encoder = (autoencoder_out_encoder.unsqueeze(2) * autoencoder_attention.unsqueeze(3)).sum(dim=0).transpose(0,1)
      autoencoder_out_full = torch.cat([autoencoder_out_decoder, autoencoder_from_encoder], dim=2)


      autoencoder_logits = self.output(self.relu(self.output_mlp(autoencoder_out_full) ))
      autoencoder_log_probs = self.logsoftmax(autoencoder_logits)

      # Prediction Loss 
      autoencoder_lossTensor = self.print_loss(autoencoder_log_probs.view(-1, len(itos)+3), target_tensor_onlyNoised[:-1].view(-1)).view(-1, NUMBER_OF_REPLICATES*args.batchSize)
      return autoencoder_lossTensor
 

  def sampleReconstructions(self, numeric, numeric_noised, NOUN, offset, numberOfBatches=args.batchSize*args.NUMBER_OF_REPLICATES, fillInBefore=-1, computeProbabilityStartingFrom=0):
      """ Draws samples from the amortized reconstruction posterior """


      input_tensor = Variable(numeric[:-1], requires_grad=False)
      input_tensor_noised = Variable(numeric_noised[:-1], requires_grad=False)
      #target_tensor = Variable(numeric_onlyNoisedOnes[1:], requires_grad=False)


      input_tensor = Variable(numeric[:-1], requires_grad=False)
      target_tensor = Variable(numeric[1:], requires_grad=False)

      input_tensor_noised = Variable(numeric_noised, requires_grad=False)


      embedded = self.word_embeddings(input_tensor)

      embedded_noised = self.word_embeddings(input_tensor_noised)

      out_encoder, _ = self.rnn_encoder(embedded_noised, None)



      hidden = None
      result  = ["" for _ in range(numberOfBatches)]
      result_numeric = [[] for _ in range(numberOfBatches)]
      embeddedLast = embedded[0].unsqueeze(0)
      amortizedPosterior = torch.zeros(numberOfBatches, device='cuda')
      zeroLogProb = torch.zeros(numberOfBatches, device='cuda')
      for i in range(args.sequence_length+1):
          out_decoder, hidden = self.rnn_decoder(embeddedLast, hidden)
#          assert embeddedLast.size()[0] == args.sequence_length-1, (embeddedLast.size()[0] , args.sequence_length)


          attention = torch.bmm(self.attention_proj(out_encoder).transpose(0,1), out_decoder.transpose(0,1).transpose(1,2))
          attention = self.attention_softmax(attention).transpose(0,1)
          from_encoder = (out_encoder.unsqueeze(2) * attention.unsqueeze(3)).sum(dim=0).transpose(0,1)
          out_full = torch.cat([out_decoder, from_encoder], dim=2)

 #         print(input_tensor.size())


          logits = self.output(self.relu(self.output_mlp(out_full) )) 
          probs = self.softmax(logits)
          if i == 15-offset:
            assert args.sequence_length == 20
            thatProbs = None #float(probs[0,:, stoi["that"]+3].mean())
#          print(i, probs[0,:, stoi["that"]+3].mean())
 #         quit()

          dist = torch.distributions.Categorical(probs=probs)
       
#          nextWord = (dist.sample())
          if i < fillInBefore:
             nextWord = numeric[i:i+1]
          else:
            sampledFromDist = dist.sample()
            logProbForSampledFromDist = dist.log_prob(sampledFromDist).squeeze(0)
 #           print(logProbForSampledFromDist.size(), numeric_noised[i].size(), zeroLogProb.size())
            assert numeric_noised.size()[0] == args.sequence_length+1
            if i < args.sequence_length: # IMPORTANT make sure the last word -- which is (due to a weird design choice) cut off -- doesn't contribute to the posterior
               amortizedPosterior += torch.where(numeric_noised[i] == 0, logProbForSampledFromDist, zeroLogProb)

            nextWord = torch.where(numeric_noised[i] == 0, sampledFromDist, numeric[i:i+1])
  #        print(nextWord.size())
          nextWordDistCPU = nextWord.cpu().numpy()[0]
          nextWordStrings = [itos_total[x] for x in nextWordDistCPU]
          for i in range(numberOfBatches):
             result[i] += " "+nextWordStrings[i]
             result_numeric[i].append( nextWordDistCPU[i] )
          embeddedLast = self.word_embeddings(nextWord)
#          print(embeddedLast.size())
      for r in result[:2]:
         print(r)
      if NOUN is not None:
         nounFraction = (float(len([x for x in result if NOUN in x]))/len(result))
         thatFraction = (float(len([x for x in result if NOUN+" that" in x]))/len(result))
      else:
         nounFraction = -1
         thatFraction = -1
      result_numeric = torch.LongTensor(result_numeric).cuda()
      assert result_numeric.size()[0] == numberOfBatches
      return result, result_numeric, (nounFraction, thatFraction), thatFraction, amortizedPosterior

    


class LanguageModel:
   """ Amortized Prediction Posterior """
   def __init__(self):
      self.rnn = torch.nn.LSTM(2*args.word_embedding_size, args.hidden_dim_lm, args.layer_num).cuda()
      self.rnn_drop = self.rnn
      self.output = torch.nn.Linear(args.hidden_dim_lm, len(itos)+3).cuda()
      self.word_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=2*args.word_embedding_size).cuda()
      self.logsoftmax = torch.nn.LogSoftmax(dim=2)
      self.train_loss = torch.nn.NLLLoss(ignore_index=0)
      self.print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
      self.char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)
      self.train_loss_chars = torch.nn.NLLLoss(ignore_index=0, reduction='sum')
      self.modules_lm = [self.rnn, self.output, self.word_embeddings]
   def forward(self, input_tensor_noised, target_tensor_full, NUMBER_OF_REPLICATES):
       lm_embedded = self.word_embeddings(input_tensor_noised)
       lm_out, lm_hidden = self.rnn_drop(lm_embedded, None)
       lm_out = lm_out[-1:]
       lm_logits = self.output(lm_out) 
       lm_log_probs = self.logsoftmax(lm_logits)
 
       # Prediction Loss 
       lm_lossTensor = self.print_loss(lm_log_probs.view(-1, len(itos)+3), target_tensor_full[-1].view(-1)).view(-1, NUMBER_OF_REPLICATES) # , args.batchSize is 1
       return lm_lossTensor 



class MemoryModel():
  """ Noise Model """
  def __init__(self):
     self.memory_mlp_inner = torch.nn.Linear(2*args.word_embedding_size, 500).cuda()
     self.memory_mlp_inner_bilinear = torch.nn.Linear(2*args.word_embedding_size, 500).cuda()
     self.memory_mlp_inner_from_pos = torch.nn.Linear(256, 500).cuda()
     self.memory_mlp_outer = torch.nn.Linear(500, 1).cuda()
     self.sigmoid = torch.nn.Sigmoid()
     self.relu = torch.nn.ReLU()
     self.positional_embeddings = torch.nn.Embedding(num_embeddings=args.sequence_length+2, embedding_dim=256).cuda()
     self.memory_word_pos_inter = torch.nn.Linear(256, 1, bias=False).cuda()
     self.memory_word_pos_inter.weight.data.fill_(0)
     self.perword_baseline_inner = torch.nn.Linear(2*args.word_embedding_size, 500).cuda()
     self.perword_baseline_outer = torch.nn.Linear(500, 1).cuda()
     self.memory_bilinear = torch.nn.Linear(256, 500, bias=False).cuda()
     self.memory_bilinear.weight.data.fill_(0)
     # Modules of the memory model
     self.modules_memory = [self.memory_mlp_inner, self.memory_mlp_outer, self.memory_mlp_inner_from_pos, self.positional_embeddings, self.perword_baseline_inner, self.perword_baseline_outer, self.memory_word_pos_inter, self.memory_bilinear, self.memory_mlp_inner_bilinear]
  def forward(self, numeric):
      embedded_everything_mem = lm.word_embeddings(numeric).detach()

      # Positional embeddings
      numeric_positions = torch.LongTensor(range(args.sequence_length+1)).cuda().unsqueeze(1)
      embedded_positions = self.positional_embeddings(numeric_positions)
      numeric_embedded = self.memory_word_pos_inter(embedded_positions)

      # Retention probabilities
      memory_byword_inner = self.memory_mlp_inner(embedded_everything_mem)
      memory_hidden_logit_per_wordtype = self.memory_mlp_outer(self.relu(memory_byword_inner))

  #    print(embedded_positions.size(), embedded_everything.size())
 #     print(self.memory_bilinear(embedded_positions).size())
#      print(self.relu(self.memory_mlp_inner_bilinear(embedded_everything.detach())).transpose(1,2).size())
      attention_bilinear_term = torch.bmm(self.memory_bilinear(embedded_positions), self.relu(self.memory_mlp_inner_bilinear(embedded_everything_mem)).transpose(1,2)).transpose(1,2)

      memory_hidden_logit = numeric_embedded + memory_hidden_logit_per_wordtype + attention_bilinear_term
      memory_hidden = self.sigmoid(memory_hidden_logit)
      return memory_hidden, embedded_everything_mem



  def compute_likelihood(self, numeric, numeric_noised, train=True, printHere=False, provideAttention=False, onlyProvideMemoryResult=False, NUMBER_OF_REPLICATES=args.NUMBER_OF_REPLICATES, expandReplicates=True, computeProbabilityStartingFrom=0):
      """ Forward pass through the entire model
        @param numeric
      """
      global hidden
      if True:
          hidden = None

      assert numeric.size() == numeric_noised.size(), (numeric.size(), numeric_noised.size())

      ######################################################
      ######################################################
      # Run Loss Model
      if expandReplicates:
         assert False
         numeric = numeric.expand(-1, NUMBER_OF_REPLICATES)
      embedded_everything_mem = lm.word_embeddings(numeric)

      # Positional embeddings
      numeric_positions = torch.LongTensor(range(args.sequence_length+1)).cuda().unsqueeze(1)
      embedded_positions = self.positional_embeddings(numeric_positions)
      numeric_embedded = self.memory_word_pos_inter(embedded_positions)

      # Retention probabilities
      memory_byword_inner = self.memory_mlp_inner(embedded_everything_mem)
      memory_hidden_logit_per_wordtype = self.memory_mlp_outer(self.relu(memory_byword_inner))

  #    print(embedded_positions.size(), embedded_everything.size())
 #     print(self.memory_bilinear(embedded_positions).size())
#      print(self.relu(self.memory_mlp_inner_bilinear(embedded_everything.detach())).transpose(1,2).size())
      attention_bilinear_term = torch.bmm(self.memory_bilinear(embedded_positions), self.relu(self.memory_mlp_inner_bilinear(embedded_everything_mem)).transpose(1,2)).transpose(1,2)

      memory_hidden_logit = numeric_embedded + memory_hidden_logit_per_wordtype + attention_bilinear_term
      memory_hidden = self.sigmoid(memory_hidden_logit)
 #     if provideAttention:
#         return memory_hidden

#      # Baseline predictions for prediction loss
 #     baselineValues = 10*self.sigmoid(self.perword_baseline_outer(self.relu(self.perword_baseline_inner(embedded_everything[-1].detach())))).squeeze(1)
  #    assert tuple(baselineValues.size()) == (NUMBER_OF_REPLICATES,)


      # NOISE MEMORY ACCORDING TO MODEL
      memory_filter = (numeric_noised != 0)
#      print(memory_filter.size(), memory_hidden.size())
      bernoulli_logprob = torch.where(memory_filter, torch.log(memory_hidden.squeeze(2)+1e-10), torch.log(1-memory_hidden.squeeze(2)+1e-10))

      punctuation = (((numeric.unsqueeze(0) == PUNCTUATION.view(12, 1, 1)).long().sum(dim=0)).bool())

      # Disregard likelihood computation on punctuation
      bernoulli_logprob = torch.where(punctuation, 0*bernoulli_logprob, bernoulli_logprob)
      # Penalize forgotten punctuation
      bernoulli_logprob = torch.where(torch.logical_and(punctuation, memory_filter==0), 0*bernoulli_logprob-10.0, bernoulli_logprob)

#      bernoulli_logprob_perBatch = bernoulli_logprob.mean(dim=0)

     # Run the following lines as a sanity check
#      print(numeric.size(), numeric_noised.size())
#      for i in range(computeProbabilityStartingFrom, bernoulli_logprob.size()[0]):
#        print(i, itos_total[int(numeric[i,0])], itos_total[int(numeric_noised[i,0])], bernoulli_logprob[i,0])


      # SPECIFICALLY FOR THIS APPLICATION (where the last element in the sequence is the first future word) CUT OFF, TO REDUCE EXTRANEOUS VARIANCE, OR POTENTIALLY PRECLUDE WEIRRD VALUES AS THAT IS ALWAYS OBLIGATORILY NOISED: I'm cutting of the final value by restricting up to -1.
      return bernoulli_logprob[computeProbabilityStartingFrom:-1].sum(dim=0)





# Build all three parts of the model
autoencoder = Autoencoder()
lm = LanguageModel()
memory = MemoryModel()

# Set up optimization

# Parameters for the retention probabilities
def parameters_memory():
   for module in memory.modules_memory:
       for param in module.parameters():
            yield param

parameters_memory_cached = [x for x in parameters_memory()]


# Set up optimization

dual_weight = torch.cuda.FloatTensor([1.0])
dual_weight.requires_grad=True

# Parameters for inference networks
def parameters_autoencoder():
   for module in autoencoder.modules_autoencoder:
       for param in module.parameters():
            yield param

def parameters_lm():
   for module in lm.modules_lm:
       for param in module.parameters():
            yield param

parameters_lm_cached = [x for x in parameters_lm()]



###############################################3

checkpoint = torch.load(glob.glob("/u/scr/mhahn/CODEBOOKS_MEMORY/*"+str(args.load_from_joint)+"*")[0])
# Load pretrained prior and amortized posteriors

# Separately load the pretrained amortized posteriors
# Amortized Reconstruction Posterior
if True or args.load_from_autoencoder is not None:
  print(checkpoint["arguments"].load_from_autoencoder)
  checkpoint_ = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+"autoencoder2_mlp_bidir_Erasure_SelectiveLoss.py"+"_code_"+str(checkpoint["arguments"].load_from_autoencoder)+".txt")
  for i in range(len(checkpoint_["components"])):
      autoencoder.modules_autoencoder[i].load_state_dict(checkpoint_["components"][i])
  del checkpoint_
 
# Amortized Prediction Posterior
if True or args.load_from_lm is not None:
  lm_file = "char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_NoChars_Erasure.py"
  checkpoint_ = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+lm_file+"_code_"+str(checkpoint["arguments"].load_from_lm)+".txt")
  for i in range(len(checkpoint_["components"])):
      lm.modules_lm[i].load_state_dict(checkpoint_["components"][i])
  del checkpoint_

from torch.autograd import Variable

if "lm_embeddings" in checkpoint:
  print(lm.word_embeddings.weight)
  assert (checkpoint["lm_embeddings"]["weight"] == lm.word_embeddings.weight).all()
  del checkpoint["lm_embeddings"]
assert set(list(checkpoint)) == set(["arguments", "words", "memory", "autoencoder"]), list(checkpoint)
assert itos == checkpoint["words"]
for i in range(len(checkpoint["memory"])):
   memory.modules_memory[i].load_state_dict(checkpoint["memory"][i])
for i in range(len(checkpoint["autoencoder"])):
   autoencoder.modules_autoencoder[i].load_state_dict(checkpoint["autoencoder"][i])




def prepareDatasetChunks(data, train=True):
      numeric = [0]
      count = 0
      print("Prepare chunks")
      numerified = []
      numerified_chars = []
      for chunk in data:
       for char in chunk:
         count += 1
         if char == ",": # Skip commas
           continue
         numerified.append((stoi[char]+3 if char in stoi else 2))

       if len(numerified) > (args.batchSize*(args.sequence_length+1)):
         sequenceLengthHere = args.sequence_length+1

         cutoff = int(len(numerified)/(args.batchSize*sequenceLengthHere)) * (args.batchSize*sequenceLengthHere)
         numerifiedCurrent = numerified[:cutoff]

         numerified = numerified[cutoff:]
       
         numerifiedCurrent = torch.LongTensor(numerifiedCurrent).view(args.batchSize, -1, sequenceLengthHere).transpose(0,1).transpose(1,2).cuda()

         numberOfSequences = numerifiedCurrent.size()[0]
         for i in range(numberOfSequences):
             yield numerifiedCurrent[i], None
       else:
         print("Skipping")











runningAverageReward = 5.0
runningAverageBaselineDeviation = 2.0
runningAveragePredictionLoss = 5.0
runningAverageReconstructionLoss = 5.0
expectedRetentionRate = 0.5


def getPunctuationMask(masks):
   assert len(masks) > 0
   if len(masks) == 1:
      return masks[0]
   else:
      punc1 = punctuation[:int(len(punctuation)/2)]
      punc2 = punctuation[int(len(punctuation)/2):]
      return torch.logical_or(getPunctuationMask(punc1), getPunctuationMask(punc2))

def product(x):
   r = 1
   for i in x:
     r *= i
   return r

# The list of tokens that the model is constrained to never erase, in order to
#  preserve information about sentence boundaries
# This also includes OOV, in order to exclude posterior samples with undefined
#  syntactic structure.
punctuation_list = [".", "OOV", '"', "(", ")", "'", '"', ":", ",", "'s", "[", "]"]
PUNCTUATION = torch.LongTensor([stoi_total[x] for x in punctuation_list]).cuda()

def forward(numeric, train=True, printHere=False, provideAttention=False, onlyProvideMemoryResult=False, NUMBER_OF_REPLICATES=args.NUMBER_OF_REPLICATES, expandReplicates=True):
      """ Forward pass through the entire model
        @param numeric
      """

      assert numeric.size()[0] == args.sequence_length+1, numeric.size()[0]
      ######################################################
      ######################################################
      # Step 1: replicate input to a batch
      if expandReplicates:
         numeric = numeric.expand(-1, NUMBER_OF_REPLICATES)

      # Input: numeric
      # Output: memory_hidden

      # Step 2: Compute retention probabilities
      memory_hidden, embedded_everything_mem = memory.forward(numeric)

      if provideAttention:
         return memory_hidden



      # Step 3: Sample representations
      memory_filter = torch.bernoulli(input=memory_hidden)
      bernoulli_logprob = torch.where(memory_filter == 1, torch.log(memory_hidden+1e-10), torch.log(1-memory_hidden+1e-10))
      bernoulli_logprob_perBatch = bernoulli_logprob.mean(dim=0)
      if args.entropy_weight > 0:
         entropy = -(memory_hidden * torch.log(memory_hidden+1e-10) + (1-memory_hidden) * torch.log(1-memory_hidden+1e-10)).mean()
      else:
         entropy=-1.0
      memory_filter = memory_filter.squeeze(2)

      # Step 4: Ensure punctuation and OOV are not deleted
      punctuation = (((numeric.unsqueeze(0) == PUNCTUATION.view(len(punctuation_list), 1, 1)).long().sum(dim=0)).bool())
        
      ####################################################################################
      numeric_noised = torch.where(torch.logical_or(punctuation, memory_filter==1), numeric, 0*numeric) #[[x if random.random() > args.deletion_rate else 0 for x in y] for y in numeric.cpu().t()]
      numeric_onlyNoisedOnes = torch.where(memory_filter == 0, numeric, 0*numeric) # target is 0 in those places where no noise has happened

      if onlyProvideMemoryResult:
        return numeric, numeric_noised
      assert False, "this version of the code is for eval only"



lossHasBeenBad = 0

import time

totalStartTime = time.time()

lastSaved = (None, None)
devLosses = []
updatesCount = 0
def showAttention(word, POS=""):
    attention = forward(torch.cuda.LongTensor([stoi[word]+3 for _ in range(args.sequence_length+1)]).view(-1, 1), train=True, printHere=True, provideAttention=True)
    attention = attention[:,0,0]
    print(*(["SCORES", word, "\t"]+[round(x,2) for x in list(attention.cpu().data.numpy())] + (["POS="+POS] if POS != "" else [])))





nounsAndVerbs = []




nounsAndVerbs.append({"item" : "238_Critical_VN1",  "compatible" : 1, "s" : "that the carpenter who the craftsman carried /confused the apprentice /was acknowledged."})
nounsAndVerbs.append({"item" : "238_Critical_VN1",  "compatible" : 2, "s" : "that the carpenter who the craftsman carried /hurt the apprentice /was acknowledged."})
nounsAndVerbs.append({"item" : "238_Critical_VN2",  "compatible" : 1, "s" : "that the daughter who the sister found /frightened the grandmother /seemed concerning."})
nounsAndVerbs.append({"item" : "238_Critical_VN2",  "compatible" : 2, "s" : "that the daughter who the sister found /greeted the grandmother /seemed concerning."})
nounsAndVerbs.append({"item" : "238_Critical_VN3",  "compatible" : 1, "s" : "that the tenant who the foreman looked for /annoyed the shepherd /proved to be made up."})
nounsAndVerbs.append({"item" : "238_Critical_VN3",  "compatible" : 2, "s" : "that the tenant who the foreman looked for /questioned the shepherd /proved to be made up."})
nounsAndVerbs.append({"item" : "238_Critical_VN5",  "compatible" : 1, "s" : "that the pharmacist who the stranger saw /distracted the customer /sounded surprising."})
nounsAndVerbs.append({"item" : "238_Critical_VN5",  "compatible" : 2, "s" : "that the pharmacist who the stranger saw /questioned the customer /sounded surprising."})
nounsAndVerbs.append({"item" : "Critical_6",        "compatible" : 1, "s" : "that the surgeon who the patient thanked /shocked his colleagues /was ridiculous."})
nounsAndVerbs.append({"item" : "Critical_6",        "compatible" : 2, "s" : "that the surgeon who the patient thanked /cured his colleagues /was ridiculous."})
nounsAndVerbs.append({"item" : "238_Critical_VAdv1","compatible" : 1, "s" : "that the commander who the president appointed /was confirmed yesterday /troubled people."})
nounsAndVerbs.append({"item" : "238_Critical_Vadv1","compatible" : 2, "s" : "that the commander who the president appointed /was fired yesterday /troubled people."})
nounsAndVerbs.append({"item" : "238_Critical_Vadv2","compatible" : 1, "s" : "that the trickster who the woman recognized /was acknowledged by the police /calmed people down."})
nounsAndVerbs.append({"item" : "238_Critical_Vadv2","compatible" : 2, "s" : "that the trickster who the woman recognized /was arrested by the police /calmed people down."})
nounsAndVerbs.append({"item" : "238_Critical_Vadv3","compatible" : 1, "s" : "that the politician who the farmer trusted /was refuted three days ago /did not bother the farmer."})
nounsAndVerbs.append({"item" : "238_Critical_Vadv3","compatible" : 2, "s" : "that the politician who the farmer trusted /was elected three days ago /did not bother the farmer."})
nounsAndVerbs.append({"item" : "238_Critical_Vadv5","compatible" : 1, "s" : "that the politician who the banker bribed /seemed credible to everyone /gave Josh the chills."})
nounsAndVerbs.append({"item" : "238_Critical_Vadv5","compatible" : 2, "s" : "that the politician who the banker bribed /seemed corrupt to everyone /gave Josh the chills."})
nounsAndVerbs.append({"item" : "238_Critical_Vadv6","compatible" : 1, "s" : "that the sculptor who the painter admired /made headlines in the US /did not surprise anyone."})
nounsAndVerbs.append({"item" : "238_Critical_Vadv6","compatible" : 2, "s" : "that the sculptor who the painter admired /made sculptures in the US /did not surprise anyone."})
nounsAndVerbs.append({"item" : "238_Critical_Vadv7","compatible" : 1, "s" : "that the runner who the psychiatrist treated /was widely known in France /turned out to be incorrect."})
nounsAndVerbs.append({"item" : "238_Critical_Vadv7","compatible" : 2, "s" : "that the runner who the psychiatrist treated /won the marathon in France /turned out to be incorrect."})
nounsAndVerbs.append({"item" : "238_Critical_Vadv8","compatible" : 1, "s" : "that the analyst who the banker trusted /appeared on TV this morning /was very believable."})
nounsAndVerbs.append({"item" : "238_Critical_Vadv8","compatible" : 2, "s" : "that the analyst who the banker trusted /repaired the TV this morning /was very believable."})
nounsAndVerbs.append({"item" : "238_Critical_Vadv9","compatible" : 1, "s" : "that the child who the medic rescued /was mentioned in newspapers /seemed very interesting."})
nounsAndVerbs.append({"item" : "238_Critical_Vadv9","compatible" : 2, "s" : "that the child who the medic rescued /wrote articles in newspapers /seemed very interesting."})
nounsAndVerbs.append({"item" : "232_Critical_0",    "compatible" : 2, "s" : "that the teacher who the principal liked /failed the student /was only a malicious smear."})
nounsAndVerbs.append({"item" : "232_Critical_0",    "compatible" : 1, "s" : "that the teacher who the principal liked /annoyed the student /was only a malicious smear."})
nounsAndVerbs.append({"item" : "232_Critical_1",    "compatible" : 2, "s" : "that the doctor who the colleague distrusted /cured the patients /seemed hard to believe."})
nounsAndVerbs.append({"item" : "232_Critical_1",    "compatible" : 1, "s" : "that the doctor who the colleague distrusted /bothered the patients /seemed hard to believe."})
nounsAndVerbs.append({"item" : "232_Critical_2",    "compatible" : 2, "s" : "that the bully who the children hated /harassed the boy /was entirely correct."})
nounsAndVerbs.append({"item" : "232_Critical_2",    "compatible" : 1, "s" : "that the bully who the children hated /shocked the boy /was entirely correct."})
nounsAndVerbs.append({"item" : "232_Critical_3",    "compatible" : 2, "s" : "that the agent who the FBI sent /arrested the criminal /was acknowledged."})
nounsAndVerbs.append({"item" : "232_Critical_3",    "compatible" : 1, "s" : "that the agent who the FBI sent /confused the criminal /was acknowledged."})
nounsAndVerbs.append({"item" : "232_Critical_4",    "compatible" : 2, "s" : "that the senator who the diplomat supported /defeated the opponent /deserved attention."})
nounsAndVerbs.append({"item" : "232_Critical_4",    "compatible" : 1, "s" : "that the senator who the diplomat supported /troubled the opponent /deserved attention."})
nounsAndVerbs.append({"item" : "232_Critical_5",    "compatible" : 2, "s" : "that the fiancé who the author met /married the bride /did not surprise anyone."})
nounsAndVerbs.append({"item" : "232_Critical_5",    "compatible" : 1, "s" : "that the fiancé who the author met /startled the bride /did not surprise anyone."})
nounsAndVerbs.append({"item" : "232_Critical_6",    "compatible" : 2, "s" : "that the businessman who the sponsor backed /fired the employee /came as a disappointment."})
nounsAndVerbs.append({"item" : "232_Critical_6",    "compatible" : 1, "s" : "that the businessman who the sponsor backed /surprised the employee /came as a disappointment."})
nounsAndVerbs.append({"item" : "232_Critical_7",    "compatible" : 2, "s" : "that the thief who the detective caught /robbed the woman /broke her family's heart."})
nounsAndVerbs.append({"item" : "232_Critical_7",    "compatible" : 1, "s" : "that the thief who the detective caught /enraged the woman /broke her family's heart."})
nounsAndVerbs.append({"item" : "232_Critical_8",    "compatible" : 2, "s" : "that the criminal who the stranger distracted /abducted the officer /seemed concerning."})
nounsAndVerbs.append({"item" : "232_Critical_8",    "compatible" : 1, "s" : "that the criminal who the stranger distracted /startled the officer /seemed concerning."})
nounsAndVerbs.append({"item" : "232_Critical_9",    "compatible" : 2, "s" : "that the customer who the vendor welcomed /contacted the clerk /was very believable."})
nounsAndVerbs.append({"item" : "232_Critical_9",    "compatible" : 1, "s" : "that the customer who the vendor welcomed /terrified the clerk /was very believable."})
nounsAndVerbs.append({"item" : "232_Critical_10",   "compatible" : 2, "s" : "that the president who the farmer admired /appointed the commander /was entirely bogus."})
nounsAndVerbs.append({"item" : "232_Critical_10",   "compatible" : 1, "s" : "that the president who the farmer admired /impressed the commander /was entirely bogus."})
nounsAndVerbs.append({"item" : "232_Critical_11",   "compatible" : 2, "s" : "that the victim who the swimmer rescued /sued the criminal /appeared on TV."})
nounsAndVerbs.append({"item" : "232_Critical_11",   "compatible" : 1, "s" : "that the victim who the swimmer rescued /surprised the criminal /appeared on TV."})
nounsAndVerbs.append({"item" : "232_Critical_12",   "compatible" : 2, "s" : "that the guest who the cousin invited /visited the uncle /drove Jill crazy."})
nounsAndVerbs.append({"item" : "232_Critical_12",   "compatible" : 1, "s" : "that the guest who the cousin invited /pleased the uncle /drove Jill crazy."})
nounsAndVerbs.append({"item" : "232_Critical_13",   "compatible" : 2, "s" : "that the psychiatrist who the nurse assisted /diagnosed the patient /became widely known."})
nounsAndVerbs.append({"item" : "232_Critical_13",   "compatible" : 1, "s" : "that the psychiatrist who the nurse assisted /horrified the patient /became widely known."})
nounsAndVerbs.append({"item" : "232_Critical_14",   "compatible" : 2, "s" : "that the driver who the guide called /phoned the tourist /was absolutely true."})
nounsAndVerbs.append({"item" : "232_Critical_14",   "compatible" : 1, "s" : "that the driver who the guide called /amazed the tourist /was absolutely true."})
nounsAndVerbs.append({"item" : "232_Critical_15",   "compatible" : 2, "s" : "that the actor who the fans loved /greeted the director /appeared to be true."})
nounsAndVerbs.append({"item" : "232_Critical_15",   "compatible" : 1, "s" : "that the actor who the fans loved /astonished the director /appeared to be true."})
nounsAndVerbs.append({"item" : "232_Critical_16",   "compatible" : 2, "s" : "that the banker who the analyst cheated /trusted the customer /proved to be made up."})
nounsAndVerbs.append({"item" : "232_Critical_16",   "compatible" : 1, "s" : "that the banker who the analyst cheated /excited the customer /proved to be made up."})
nounsAndVerbs.append({"item" : "232_Critical_17",   "compatible" : 2, "s" : "that the judge who the attorney hated /acquitted the defendant /was a lie."})
nounsAndVerbs.append({"item" : "232_Critical_17",   "compatible" : 1, "s" : "that the judge who the attorney hated /vindicated the defendant /was a lie."})
nounsAndVerbs.append({"item" : "232_Critical_18",   "compatible" : 2, "s" : "that the captain who the crew trusted /commanded the sailor /was nice to hear."})
nounsAndVerbs.append({"item" : "232_Critical_18",   "compatible" : 1, "s" : "that the captain who the crew trusted /motivated the sailor /was nice to hear."})
nounsAndVerbs.append({"item" : "232_Critical_19",   "compatible" : 2, "s" : "that the manager who the boss authorized /hired the intern /seemed absurd."})
nounsAndVerbs.append({"item" : "232_Critical_19",   "compatible" : 1, "s" : "that the manager who the boss authorized /saddened the intern /seemed absurd."})
nounsAndVerbs.append({"item" : "232_Critical_20",   "compatible" : 2, "s" : "that the plaintiff who the jury interrogated /interrupted the witness /made it into the news."})
nounsAndVerbs.append({"item" : "232_Critical_20",   "compatible" : 1, "s" : "that the plaintiff who the jury interrogated /startled the witness /made it into the news."})
nounsAndVerbs.append({"item" : "232_Critical_21",   "compatible" : 2, "s" : "that the drinker who the thug hit /defeated the bartender /sounded hilarious."})
nounsAndVerbs.append({"item" : "232_Critical_21",   "compatible" : 1, "s" : "that the drinker who the thug hit /stunned the bartender /sounded hilarious."})
nounsAndVerbs.append({"item" : "232_Critical_23",   "compatible" : 2, "s" : "that the medic who the survivor thanked /greeted the surgeon /turned out to be untrue."})
nounsAndVerbs.append({"item" : "232_Critical_23",   "compatible" : 1, "s" : "that the medic who the survivor thanked /surprised the surgeon /turned out to be untrue."})
nounsAndVerbs.append({"item" : "232_Critical_24",   "compatible" : 2, "s" : "that the lifeguard who the soldier taught /rescued the swimmer /took the townspeople by surprise."})
nounsAndVerbs.append({"item" : "232_Critical_24",   "compatible" : 1, "s" : "that the lifeguard who the soldier taught /encouraged the swimmer /took the townspeople by surprise."})
nounsAndVerbs.append({"item" : "232_Critical_25",   "compatible" : 2, "s" : "that the fisherman who the gardener helped /admired the politician /was interesting."})
nounsAndVerbs.append({"item" : "232_Critical_25",   "compatible" : 1, "s" : "that the fisherman who the gardener helped /delighted the politician /was interesting."})
nounsAndVerbs.append({"item" : "232_Critical_26",   "compatible" : 2, "s" : "that the janitor who the organizer criticized /ignored the audience /was funny."})
nounsAndVerbs.append({"item" : "232_Critical_26",   "compatible" : 1, "s" : "that the janitor who the organizer criticized /amused the audience /was funny."})
nounsAndVerbs.append({"item" : "232_Critical_27",   "compatible" : 2, "s" : "that the investor who the scientist hated /deceived the entrepreneur /drove everyone crazy."})
nounsAndVerbs.append({"item" : "232_Critical_27",   "compatible" : 1, "s" : "that the investor who the scientist hated /disappointed the entrepreneur /drove everyone crazy."})
nounsAndVerbs.append({"item" : "232_Critical_28",   "compatible" : 2, "s" : "that the firefighter who the neighbor insulted /rescued the homeowner /went unnoticed."})
nounsAndVerbs.append({"item" : "232_Critical_28",   "compatible" : 1, "s" : "that the firefighter who the neighbor insulted /discouraged the homeowner /went unnoticed."})
nounsAndVerbs.append({"item" : "232_Critical_30",   "compatible" : 2, "s" : "that the plumber who the apprentice consulted /assisted the woman /was true."})
nounsAndVerbs.append({"item" : "232_Critical_30",   "compatible" : 1, "s" : "that the plumber who the apprentice consulted /puzzled the woman /was true."})





for x in nounsAndVerbs:
# for z in x:
   for q in x["s"].split(" "):
    q = q.strip("/").strip(".").lower()
    if q not in stoi_total and q not in ["X", "Y", "XXXX"]:
     print("OOV WARNING", "#"+q+"#")
#quit()


#nounsAndVerbs.append(["the senator",        "the diplomat",       "opposed"])

#nounsAndVerbs = nounsAndVerbs[:1]

topNouns = []
topNouns.append("report")
topNouns.append("story")       
#topNouns.append("disclosure")
topNouns.append("proof")
topNouns.append("confirmation")  
topNouns.append("information")
topNouns.append("evidence")
topNouns.append("reminder")
topNouns.append("rumor")
#topNouns.append("thought")
topNouns.append("suggestion")
topNouns.append( "revelation")  
topNouns.append( "belief")
topNouns.append( "fact")
topNouns.append( "realization")
topNouns.append( "suspicion")
topNouns.append( "certainty")
topNouns.append( "idea")
topNouns.append( "admission") 
topNouns.append( "confirmation")
topNouns.append( "complaint"    )
topNouns.append( "certainty"   )
topNouns.append( "prediction"  )
topNouns.append( "declaration")
topNouns.append( "proof"   )
topNouns.append( "suspicion")    
topNouns.append( "allegation"   )
topNouns.append( "revelation"   )
topNouns.append( "realization")
topNouns.append( "news")
topNouns.append( "opinion" )
topNouns.append( "idea")
topNouns.append("myth")

topNouns.append("announcement")
topNouns.append("suspicion")
topNouns.append("allegation")
topNouns.append("realization")
topNouns.append("indication")
topNouns.append("remark")
topNouns.append("speculation")
topNouns.append("assurance")
topNouns.append("presumption")
topNouns.append("concern")
topNouns.append("finding")
topNouns.append("assertion")
topNouns.append("feeling")
topNouns.append("perception")
topNouns.append("statement")
topNouns.append("assumption")
topNouns.append("conclusion")


topNouns.append("report")
topNouns.append("story")
#topNouns.append("disclosure")
topNouns.append("confirmation")   
topNouns.append("information")
topNouns.append("evidence")
topNouns.append("reminder")
topNouns.append("rumor")
topNouns.append("thought")
topNouns.append("suggestion")
topNouns.append("revelation")    
topNouns.append("belief")
#topNouns.append("inkling") # this is OOV for the model
topNouns.append("suspicion")
topNouns.append("idea")
topNouns.append("claim")
topNouns.append("news")
topNouns.append("proof")
topNouns.append("admission")
topNouns.append("declaration")

topNouns.append("assessment")
topNouns.append("truth")
topNouns.append("declaration")
topNouns.append("complaint")
topNouns.append("admission")
topNouns.append("disclosure")
topNouns.append("confirmation")
topNouns.append("guess")
topNouns.append("remark")
topNouns.append("news")
topNouns.append("proof")
topNouns.append("message")
topNouns.append("announcement")
topNouns.append("statement")
topNouns.append("thought")
topNouns.append("allegation")
topNouns.append("indication")
topNouns.append("recognition")
topNouns.append("speculation")
topNouns.append("accusation")
topNouns.append("reminder")
topNouns.append("rumor")
topNouns.append("finding")
topNouns.append("idea")
topNouns.append("feeling")
topNouns.append("conjecture")
topNouns.append("perception")
topNouns.append("certainty")
topNouns.append("revelation")
topNouns.append("understanding")
topNouns.append("claim")
topNouns.append("view")
topNouns.append("observation")
topNouns.append("conviction")
topNouns.append("presumption")
topNouns.append("intuition")
topNouns.append("opinion")
topNouns.append("conclusion")
topNouns.append("notion")
topNouns.append("suggestion")
topNouns.append("sense")
topNouns.append("suspicion")
topNouns.append("assurance")
topNouns.append("insinuation")
topNouns.append("realization")
topNouns.append("assertion")
topNouns.append("impression")
topNouns.append("contention")
topNouns.append("assumption")
topNouns.append("belief")
topNouns.append("fact")

topNouns = list(set(topNouns))



with open("../../../../forgetting/corpus_counts/wikipedia/results/counts4NEW_Processed.tsv", "r") as inFile:
   counts = [x.replace('"', '').split("\t") for x in inFile.read().strip().split("\n")]
   header = ["LineNum"] + counts[0]
   assert len(header) == len(counts[1])
   header = dict(list(zip(header, range(len(header)))))
   counts = {line[header["Noun"]] : line for line in counts[1:]}


print(len(topNouns))
print([x for x in topNouns if x not in counts])
topNouns = [x for x in topNouns if x in counts]

def thatBias(noun):
   return math.log(float(counts[noun][header["CountThat"]]))-math.log(float(counts[noun][header["CountBare"]]))

topNouns = sorted(list(set(topNouns)), key=lambda x:thatBias(x))

print(topNouns)
print(len(topNouns))
#quit()


# This is to ensure the tsv files are useful even when the script is stopped prematurely
random.shuffle(topNouns)

    


# Helper Functions

def correlation(x, y):
   variance_x = (x.pow(2)).mean() - x.mean().pow(2)
   variance_y = (y.pow(2)).mean() - y.mean().pow(2)
   return ((x-x.mean())* (y-y.mean())).mean()/(variance_x*variance_y).sqrt()


def rindex(x, y):
   return max([i for i in range(len(x)) if x[i] == y])


def encodeContextCrop(inp, context):
     sentence = context.strip() + " " + inp.strip()
     print("ENCODING", sentence)
     numerified = [stoi_total[char] if char in stoi_total else 2 for char in sentence.split(" ")]
     print(len(numerified))
     numerified = numerified[-args.sequence_length-1:]
     numerified = torch.LongTensor([numerified for _ in range(args.batchSize)]).t().cuda()
     return numerified

def flatten(x):
   l = []
   for y in x:
     for z in y:
        l.append(z)
   return l







def divideDicts(y, z):
   r = {}
   for x in y:
     r[x] = y[x]/z[x]
   return r

def getTotalSentenceSurprisals(SANITY="Model", VERBS=2): # Surprisal for EOS after 2 or 3 verbs
    assert SANITY in ["ModelTmp", "Model", "Sanity", "ZeroLoss"]
    assert VERBS in [1,2]
#    print(plain_lm) 
    surprisalsPerNoun = {}
    surprisalsReweightedPerNoun = {}
    thatFractionsPerNoun = {}
    thatFractionsReweightedPerNoun = {}
    numberOfSamples = 12
    import scoreWithGPT2Medium as scoreWithGPT2
    global topNouns
#    topNouns = ["fact", "report"]
    outFilePath = "/u/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/"+__file__+"_"+str(args.load_from_joint)+"_"+SANITY
    if len(glob.glob(outFilePath)) > 0:
        mode = "a"
        nounsDone = set()
        with open(outFilePath, "r") as inFile:
           for line in inFile:
              try:
                 noun_ = line[:line.index("\t")]
              except ValueError:
               print("EMPTY LINE?", line, file=sys.stderr)
              finally:
                 nounsDone.add(noun_)
    else:
        nounsDone = set()
        mode = "w"
    print("NOUNS DONE", nounsDone, file=sys.stderr)
    with open(outFilePath, mode) if SANITY != "ModelTmp" else sys.stdout as outFile:
     if mode == "w":
        print("\t".join(["Noun", "Item", "Region", "Condition", "Surprisal", "SurprisalReweighted", "ThatFraction", "ThatFractionReweighted", "SurprisalsWithThat", "SurprisalsWithoutThat", "Word"]), file=outFile)
     with torch.no_grad():
      TRIALS_COUNT = 0
      TOTAL_TRIALS = len(topNouns) * len(nounsAndVerbs) * 2 * 1
      for nounIndex, NOUN in enumerate(topNouns):
        if NOUN in nounsDone:
           continue  
        print(NOUN, "Time:", time.time() - startTimePredictions, nounIndex/len(topNouns), file=sys.stderr)
        thatFractions = {x : defaultdict(float) for x in ["SC_compatible", "NoSC_neither", "SC_incompatible", "SCRC_compatible", "SCRC_incompatible"]}
        thatFractionsReweighted = {x : defaultdict(float) for x in ["SC_compatible", "NoSC_neither", "SC_incompatible", "SCRC_compatible", "SCRC_incompatible"]}
        thatFractionsCount = {x : defaultdict(float) for x in ["SC_compatible", "NoSC_neither", "SC_incompatible", "SCRC_compatible", "SCRC_incompatible"]}
        surprisalReweightedByRegions = {x : defaultdict(float) for x in ["SC_compatible", "NoSC_neither", "SC_incompatible", "SCRC_compatible", "SCRC_incompatible"]}
        surprisalByRegions = {x : defaultdict(float) for x in ["SC_compatible", "NoSC_neither", "SC_incompatible", "SCRC_compatible", "SCRC_incompatible"]}
        surprisalCountByRegions = {x : defaultdict(float) for x in ["SC_compatible", "NoSC_neither", "SC_incompatible", "SCRC_compatible", "SCRC_incompatible"]}
        itemIDs = set()
        for sentenceID in range(len(nounsAndVerbs)):
          print(sentenceID)
          context = None
          for condition in ["SCRC", "SC","NoSC"]:
            TRIALS_COUNT += 1
            print("TRIALS", TRIALS_COUNT/TOTAL_TRIALS)
            sentenceListDict = nounsAndVerbs[sentenceID]
            itemID = sentenceListDict["item"]
            compatible = [None, "compatible", "incompatible"][sentenceListDict["compatible"]] if condition != "NoSC" else "neither"
            sentence1, V2, V3 = sentenceListDict["s"].lower().split("/")
            sentence1, V1 = sentence1.split(" who ")
            sentenceList = [(" "+sentence1.strip()).replace(" that ", "").strip(), V1.strip(), V2.strip().strip("/").strip("."), V3.strip().strip("/").strip(".")]                                  
            assert len(sentenceList) >= 4, sentenceList
            if condition == "NoSC" and itemID in itemIDs:
               continue
            if condition == "SC":
               context = f"the {NOUN} that {sentenceList[0]}"
               regionsToDo = [(sentenceList[2], "V2"), (sentenceList[3].split(" ")[0], "V1")]
               remainingInput = flatten([x[0].split(" ") for x in regionsToDo])
               regions = flatten([[f"{region}_{c}" for c, _ in enumerate(words.split(" "))] for words, region in regionsToDo])
               assert len(remainingInput) == len(regions), (regionsToDo, remainingInput, regions)
            elif condition == "NoSC":
               itemIDs.add(itemID)
               context = f"the {NOUN}"
               regionsToDo = [(sentenceList[3].split(" ")[0], "V1")]
               remainingInput = flatten([x[0].split(" ") for x in regionsToDo])
               regions = flatten([[f"{region}_{c}" for c, _ in enumerate(words.split(" "))] for words, region in regionsToDo])
               assert len(remainingInput) == len(regions), (regionsToDo, remainingInput, regions)
            elif condition == "SCRC":
               context = f"the {NOUN} that {sentenceList[0]} who {sentenceList[1]}"
               regionsToDo = [(sentenceList[2], "V2"), (sentenceList[3].split(" ")[0], "V1")]
               remainingInput = flatten([x[0].split(" ") for x in regionsToDo])
               regions = flatten([[f"{region}_{c}" for c, _ in enumerate(words.split(" "))] for words, region in regionsToDo])
               assert len(remainingInput) == len(regions), (regionsToDo, remainingInput, regions)
            else:
               assert False
            print("INPUT", context, remainingInput)
            assert len(remainingInput) > 0
            for i in range(len(remainingInput)):
              if regions[i] not in ["V1_0"]: #.startswith("V2"):
                continue
              numerified = encodeContextCrop(" ".join(remainingInput[:i+1]), "later the nurse suggested they treat the patient with an antibiotic but in the end this did not happen . " + context)
              pointWhereToStart = args.sequence_length - len(context.split(" ")) - i - 1
              assert pointWhereToStart >= 0, (args.sequence_length, i, len(context.split(" ")))
              assert numerified.size()[0] == args.sequence_length+1, (numerified.size())
              # Run the memory model. We collect 'numberOfSamples' many replicates.
              if SANITY == "Sanity":
                 numeric = numerified
                 numeric = numeric.expand(-1, numberOfSamples)
                 numeric_noised = torch.where(numeric == stoi["that"]+3, 0*numeric, numeric)
              elif SANITY == "ZeroLoss":
                 numeric = numerified
                 numeric = numeric.expand(-1, numberOfSamples)
                 numeric_noised = numeric
              else:
                 assert SANITY in ["Model", "ModelTmp"]
                 numeric, numeric_noised = forward(numerified, train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True, NUMBER_OF_REPLICATES=numberOfSamples)
                 numeric_noised = torch.where(numeric == stoi["."]+3, numeric, numeric_noised)
              # Next, expand the tensor to get 24 samples from the reconstruction posterior for each replicate
              numeric = numeric.unsqueeze(2).expand(-1, -1, 24).view(-1, numberOfSamples*24)
              numeric_noised = numeric_noised.unsqueeze(2).expand(-1, -1, 24).contiguous().view(-1, numberOfSamples*24)
              numeric_noised[args.sequence_length] = 0 # A simple hack for dealing with the issue that the last word 
              # Now get samples from the amortized reconstruction posterior
              print("NOISED: ", " ".join([itos_total[int(x)] for x in numeric_noised[:,0].cpu()]))
              result, resultNumeric, fractions, thatProbs, amortizedPosterior = autoencoder.sampleReconstructions(numeric, numeric_noised, NOUN, 2, numberOfBatches=numberOfSamples*24, fillInBefore=pointWhereToStart)
              # get THAT fractions
              if "NoSC" not in condition: # and i == 0:
                 resultNumericPrevious = resultNumeric
                 locationThat = context.split(" ")[::-1].index("that")+i+2
                 thatFractionHere = float((resultNumeric[:, -locationThat] == stoi_total["that"]).float().mean())
                 thatFractions[condition+"_"+compatible][regions[i]]+=thatFractionHere
                 thatFractionsCount[condition+"_"+compatible][regions[i]]+=1
              else:
                 thatFractionHere = -1

              resultNumeric = resultNumeric.transpose(0,1).contiguous()


              likelihood = memory.compute_likelihood(resultNumeric, numeric_noised, train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True, NUMBER_OF_REPLICATES=1, computeProbabilityStartingFrom=pointWhereToStart, expandReplicates=False)




              nextWord = torch.LongTensor([stoi_total.get(remainingInput[i], stoi_total["OOV"]) for _ in range(numberOfSamples*24)]).unsqueeze(0).cuda()
              resultNumeric = torch.cat([resultNumeric[:-1], nextWord], dim=0).contiguous()
              # Evaluate the prior on these samples to estimate next-word surprisal

              resultNumeric_cpu = resultNumeric.detach().cpu()
              batch = [" ".join([itos_total[resultNumeric_cpu[r,s]] for r in range(pointWhereToStart+1, resultNumeric.size()[0])]) for s in range(resultNumeric.size()[1])]
              for h in range(len(batch)):
                 batch[h] = batch[h][:1].upper() + batch[h][1:]
                 assert batch[h][0] != " ", batch[h]
#              print(batch)
              totalSurprisal = scoreWithGPT2.scoreSentences(batch)
              surprisals_past = torch.FloatTensor([x["past"] for x in totalSurprisal]).cuda().view(numberOfSamples, 24)
              surprisals_nextWord = torch.FloatTensor([x["next"] for x in totalSurprisal]).cuda().view(numberOfSamples, 24)

             # Here is alternative code using a plain LSTM LM instead of GPT-2
#              totalSurprisal, _, samplesFromLM, predictionsPlainLM = plain_lm.forward(resultNumeric, train=False, computeSurprisals=True, returnLastSurprisal=False, numberOfBatches=numberOfSamples*24)
#              assert resultNumeric.size()[0] == args.sequence_length+1
#              assert totalSurprisal.size()[0] == args.sequence_length
#              # For each of the `numberOfSamples' many replicates, evaluate (i) the probability of the next word under the Monte Carlo estimate of the next-word posterior, (ii) the corresponding surprisal, (iii) the average of those surprisals across the 'numberOfSamples' many replicates.
#              totalSurprisal = totalSurprisal.view(args.sequence_length, numberOfSamples, 24)
#              surprisals_past = totalSurprisal[:-1].sum(dim=0)
#              surprisals_nextWord = totalSurprisal[-1]

              # where numberOfSamples is how many samples we take from the noise model, and 24 is how many samples are drawn from the amortized posterior for each noised sample
              amortizedPosterior = amortizedPosterior.view(numberOfSamples, 24)
              likelihood = likelihood.view(numberOfSamples, 24)
              unnormalizedLogTruePosterior = likelihood - surprisals_past
              assert float(unnormalizedLogTruePosterior.max()) <= 1e-5
              assert float(amortizedPosterior.max()) <= 1e-5
              log_importance_weights = unnormalizedLogTruePosterior - amortizedPosterior
              log_importance_weights_maxima, _ = log_importance_weights.max(dim=1, keepdim=True)
              print(log_importance_weights[0])
              for j in range(24): # TODO the importance weights seem wacky
                 if j % 3 != 0:
                    continue
                 print(j, "@@", result[j], float(surprisals_past[0, j]), float(surprisals_nextWord[0, j]), float(log_importance_weights[0, j]), float(likelihood[0, j]), float(amortizedPosterior[0, j]))
              print(" ".join([itos_total[int(x)] for x in numeric_noised[:, 0].detach().cpu()]))
              log_importance_weighted_probs_unnormalized = torch.log(torch.exp(log_importance_weights - log_importance_weights_maxima - surprisals_nextWord).sum(dim=1)) + log_importance_weights_maxima.squeeze(1)
              log_importance_weights_sum = torch.log(torch.exp(log_importance_weights - log_importance_weights_maxima).sum(dim=1)) + log_importance_weights_maxima.squeeze(1)
              reweightedSurprisals = -(log_importance_weighted_probs_unnormalized - log_importance_weights_sum)
              reweightedSurprisalsMean = reweightedSurprisals.mean()

              surprisalOfNextWord = surprisals_nextWord.exp().mean(dim=1).log().mean()
              nextWordSurprisal_cpu = surprisals_nextWord.view(-1).detach().cpu()

              if "NoSC" not in condition: # and i == 0:
                 surprisalsWithThat = float(surprisals_nextWord[(resultNumericPrevious[:, -locationThat] == stoi_total["that"]).view(-1, 24)].mean())
                 surprisalsWithoutThat = float(surprisals_nextWord[(resultNumericPrevious[:, -locationThat] != stoi_total["that"]).view(-1, 24)].mean())
                 print("Surp with and without that", surprisalsWithThat, surprisalsWithoutThat)               
                 thatFractionReweightedHere = float((((resultNumericPrevious[:, -locationThat] == stoi_total["that"]).float().view(-1, 24) * torch.exp(log_importance_weights - log_importance_weights_sum.unsqueeze(1))).sum(dim=1)).mean())
                 thatFractionsReweighted[condition+"_"+compatible][regions[i]]+=thatFractionReweightedHere
              else:
                 thatFractionReweightedHere = -1


              for q in range(0, min(3*24, resultNumeric.size()[1]),  24):
                  print("DENOISED PREFIX + NEXT WORD", " ".join([itos_total[int(x)] for x in resultNumeric[:,q]]), float(nextWordSurprisal_cpu[q])) #, float(reweightedSurprisal_cpu[q//24]))
              print("SURPRISAL", NOUN, sentenceList[0], condition+"_"+compatible, i, regions[i], remainingInput[i],float( surprisalOfNextWord), float(reweightedSurprisalsMean))
              surprisalReweightedByRegions[condition+"_"+compatible][regions[i]] += float( reweightedSurprisalsMean)
              surprisalByRegions[condition+"_"+compatible][regions[i]] += float( surprisalOfNextWord)
              surprisalCountByRegions[condition+"_"+compatible][regions[i]] += 1

              print("\t".join([str(w) for w in [NOUN, itemID, regions[i], condition+"_"+compatible[:2], round(float( surprisalOfNextWord),3), round(float( reweightedSurprisalsMean),3), int(100*thatFractionHere), int(100*thatFractionReweightedHere), surprisalsWithThat, surprisalsWithoutThat, remainingInput[i]]]), file=outFile)


        print(surprisalByRegions)
        print(surprisalReweightedByRegions)
        print(thatFractions)
        print("NOUNS SO FAR", topNouns.index(NOUN))
        assert NOUN not in surprisalsPerNoun # I think that in previous versions of these scripts the indentation was wrong, and this was overwitten multiple times
        assert NOUN not in surprisalsReweightedPerNoun # I think that in previous versions of these scripts the indentation was wrong, and this was overwitten multiple times
        print(surprisalByRegions)
        surprisalsReweightedPerNoun[NOUN] = {x : divideDicts(surprisalReweightedByRegions[x], surprisalCountByRegions[x]) for x in surprisalReweightedByRegions}
        surprisalsPerNoun[NOUN] = {x : divideDicts(surprisalByRegions[x], surprisalCountByRegions[x]) for x in surprisalByRegions}
        thatFractionsReweightedPerNoun[NOUN] = {x : divideDicts(thatFractionsReweighted[x], thatFractionsCount[x]) for x in thatFractionsReweighted}
        thatFractionsPerNoun[NOUN] = {x : divideDicts(thatFractions[x], thatFractionsCount[x]) for x in thatFractions}
        print(thatFractionsPerNoun[NOUN])
    print("SURPRISALS BY NOUN", surprisalsPerNoun)
    print("THAT (fixed) BY NOUN", thatFractionsPerNoun)
    print("SURPRISALS_PER_NOUN PLAIN_LM, WITH VERB, NEW")
    outFilePath = "/u/scr/mhahn/reinforce-logs-both-short/full-logs-tsv/"+__file__+"_"+str(args.load_from_joint)+"_"+SANITY
    if len(glob.glob(outFilePath)) == 0:
     with open(outFilePath, "w")  if SANITY != "ModelTmp" else sys.stdout as outFile:
      print("Noun", "Region", "Condition", "Surprisal", "SurprisalReweighted", "ThatFraction", "ThatFractionReweighted", file=outFile)
      for noun in topNouns:
       for condition in surprisalsPerNoun[noun]:
         for region in surprisalsPerNoun[noun][condition]:
           print(noun, region, condition, surprisalsPerNoun[noun][condition][region], surprisalsReweightedPerNoun[noun][condition][region], thatFractionsPerNoun[noun][condition][region] if "NoSC" not in condition else "NA", thatFractionsReweightedPerNoun[noun][condition][region] if "NoSC" not in condition else "NA", file=outFile)
    # For sanity-checking: Prints correlations between surprisal and that-bias
    for region in ["V2_0", "V2_1", "V1_0"]:
      for condition in surprisalsPerNoun["fact"]:
       if region not in surprisalsPerNoun["fact"][condition]:
          continue
       print(SANITY, condition, "CORR", region, correlation(torch.FloatTensor([thatBias(x) for x in topNouns]), torch.FloatTensor([surprisalsPerNoun[x][condition][region] for x in topNouns])), correlation(torch.FloatTensor([thatBias(x) for x in topNouns]), torch.FloatTensor([surprisalsReweightedPerNoun[x][condition][region] for x in topNouns])), correlation(torch.FloatTensor([thatBias(x) for x in topNouns]), torch.FloatTensor([thatFractionsPerNoun[x][condition][region] for x in topNouns])) if "NoSC" not in condition else 0 , correlation(torch.FloatTensor([thatBias(x) for x in topNouns]), torch.FloatTensor([thatFractionsReweightedPerNoun[x][condition][region] for x in topNouns])) if "NoSC" not in condition else 0 )
       surprisals = torch.FloatTensor([surprisalsPerNoun[x][condition][region] for x in topNouns])
       print(condition, surprisals.mean(), "SD", math.sqrt(surprisals.pow(2).mean() - surprisals.mean().pow(2)))




startTimePredictions = time.time()



startTimeTotal = time.time()
startTimePredictions = time.time()
startTimeTotal = time.time()

if True:
       with open("/u/scr/mhahn/reinforce-logs-both-short/full-logs/"+__file__+"_"+str(args.load_from_joint), "w") as outFile:
         print(updatesCount, "Slurm", os.environ["SLURM_JOB_ID"], file=outFile)
         print(checkpoint["arguments"], file=outFile)


       
       # Record reconstructions and surprisals
       with open("/u/scr/mhahn/reinforce-logs-both-short/full-logs/"+__file__+"_"+str(args.load_from_joint), "w") as outFile:
         startTimePredictions = time.time()

         sys.stdout = outFile
         print(updatesCount, "Slurm", os.environ["SLURM_JOB_ID"])
         print(checkpoint["arguments"])
         print("=========================")
         showAttention("the")
         showAttention("was")
         showAttention("that")
         showAttention("fact")
         showAttention("information")
         showAttention("report")
         showAttention("belief")
         showAttention("finding")
         showAttention("prediction")
         showAttention("of")
         showAttention("by")
         showAttention("about")
         getTotalSentenceSurprisals(SANITY="Model")


         print("=========================")
         # Determiner
         showAttention("the", POS="Det")
         showAttention("a", POS="Det")
         # Verbs
         showAttention("was")
         showAttention("pleased", POS="Verb")
         showAttention("invited", POS="Verb")
         showAttention("annoyed", POS="Verb")
         showAttention("did", POS="Verb")
         showAttention("failed", POS="Verb")
         showAttention("trusted", POS="Verb")
         showAttention("bothered", POS="Verb")
         showAttention("admired", POS="Verb")
         showAttention("impressed", POS="Verb")
         showAttention("shocked", POS="Verb")
         showAttention("appointed", POS="Verb")
         showAttention("supported", POS="Verb")
         showAttention("looked", POS="Verb")
         # that
         showAttention("that", POS="that")
         # Noun
         showAttention("fact", POS="Verb")
         showAttention("information", POS="Verb")
         showAttention("report", POS="Noun")
         showAttention("belief", POS="Noun")
         showAttention("finding", POS="Noun")
         showAttention("prediction", POS="Noun")
         showAttention("musician", POS="Noun")
         showAttention("surgeon", POS="Noun")
         showAttention("survivor", POS="Noun")
         showAttention("guide", POS="Noun")
         showAttention("fans", POS="Noun")
         showAttention("sponsor", POS="Noun")
         showAttention("detective", POS="Noun")
         showAttention("time", POS="Noun")
         showAttention("years", POS="Noun")
         showAttention("name", POS="Noun")
         showAttention("country", POS="Noun")
         showAttention("school", POS="Noun")
         showAttention("agreement", POS="Noun")
         showAttention("series", POS="Noun")
         showAttention("producers", POS="Noun")
         showAttention("concerts", POS="Noun")
         showAttention("classification", POS="Noun")
         showAttention("house", POS="Noun")
         showAttention("circle", POS="Noun")
         showAttention("balance", POS="Noun")
         showAttention("cartoon", POS="Noun")
         showAttention("dancers", POS="Noun")
         showAttention("immigrant", POS="Noun")
         showAttention("teacher", POS="Noun")
         showAttention("doctor", POS="Noun")
         showAttention("patient", POS="Noun")
         # Preposition
         showAttention("of", POS="Prep")
         showAttention("for", POS="Prep")
         showAttention("to", POS="Prep")
         showAttention("in", POS="Prep")
         showAttention("by", POS="Prep")
         showAttention("about", POS="Prep")
         # Pronouns
         showAttention("you", POS="Pron")
         showAttention("we", POS="Pron")
         showAttention("he", POS="Pron")
         showAttention("she", POS="Pron")
         sys.stdout = STDOUT


print("=========================")
showAttention("the")
showAttention("was")
showAttention("that")
showAttention("fact")
showAttention("information")
showAttention("report")
showAttention("belief")
showAttention("finding")
showAttention("prediction")
showAttention("of")
showAttention("by")
showAttention("about")


