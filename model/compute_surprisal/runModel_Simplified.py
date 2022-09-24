# Running the model on your own dataset:

# (1) Replace calibrationSentences with the sentences of interest. Currently, this list consists of the fillers.
# (2) Replace OUTPUT_PATH with the correct path for storing the output.
# (3) Run thecorresponding  RUNALL script to get predictions across model runs.

# This is derived from the corresponding script for the fillers.
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

# Replace these sentences with the sentences of interest
calibrationSentences = []

calibrationSentences.append("The divorcee has come to love her life ever since she got divorced.") 
calibrationSentences.append("The mathematician at the banquet baffled the philosopher although she rarely needed anyone else's help.")
calibrationSentences.append("The showman travels to different cities every month.")
calibrationSentences.append("The roommate takes out the garbage every week.")
calibrationSentences.append("The dragon wounded the knight although he was far too crippled to protect the princess.")
calibrationSentences.append("The office-worker worked through the stack of files on his desk quickly.")
calibrationSentences.append("The firemen at the scene apprehended the arsonist because there was a great deal of evidence pointing to his guilt.")
calibrationSentences.append("During the season, the choir holds rehearsals in the church regularly.")
calibrationSentences.append("The speaker who the historian offended kicked a chair after the talk was over and everyone had left the room.")
calibrationSentences.append("The milkman punctually delivers the milk at the door every day.")
calibrationSentences.append("The quarterback dated the cheerleader although this hurt her reputation around school.")
calibrationSentences.append("The citizens of France eat oysters.")
calibrationSentences.append("The bully punched the kid after all the kids had to leave to go to class.")
calibrationSentences.append("After the argument, the husband ignored his wife.")
calibrationSentences.append("The engineer who the lawyer who was by the elevator scolded blamed the secretary but nobody listened to his complaints.")
calibrationSentences.append("The librarian put the book onto the shelf.")
calibrationSentences.append("The photographer processed the film on time.")
calibrationSentences.append("The spider that the boy who was in the yard captured scared the dog since it was larger than the average spider.")
calibrationSentences.append("The sportsman goes jogging in the park regularly.")
calibrationSentences.append("The customer who was on the phone contacted the operator because the new long-distance pricing plan was extremely inconvenient.")
calibrationSentences.append("The private tutor explained the assignment carefully.")
calibrationSentences.append("The audience who was at the club booed the singer before the owner of the bar could remove him from the stage.")
calibrationSentences.append("The defender is constantly scolding the keeper.")
calibrationSentences.append("The hippies who the police at the concert arrested complained to the officials while the last act was going on stage.")
calibrationSentences.append("The natives on the island captured the anthropologist because she had information that could help the tribe.")
calibrationSentences.append("The trainee knew that the task which the director had set for him was impossible to finish within a week.")
calibrationSentences.append("The administrator who the nurse from the clinic supervised scolded the medic while a patient was brought into the emergency room.")
calibrationSentences.append("The company was sure that its new product, which its researchers had developed, would soon be sold out.")
calibrationSentences.append("The astronaut that the journalists who were at the launch worshipped criticized the administrators after he discovered a potential leak in the fuel tank.")
calibrationSentences.append("The janitor who the doorman who was at the hotel chatted with bothered a guest but the manager decided not to fire him for it.")
calibrationSentences.append("The technician at the show repaired the robot while people were taking a break for coffee.")
calibrationSentences.append("The salesman feared that the printer which the customer bought was damaged.")
calibrationSentences.append("The students studied the surgeon whenever he performed an important operation.")
calibrationSentences.append("The locksmith can crack the safe easily.")
calibrationSentences.append("The woman who was in the apartment hired the plumber despite the fact that he couldn't fix the toilet.")
calibrationSentences.append("Yesterday the swimmer saw only a turtle at the beach.")
calibrationSentences.append("The surgeon who the detective who was on the case consulted questioned the coroner because the markings on the body were difficult to explain.")
calibrationSentences.append("The gangster who the detective at the club followed implicated the waitress because the police suspected he had murdered the shopkeeper.")
calibrationSentences.append("During the party everybody was dancing to rock music.")
calibrationSentences.append("The fans at the concert loved the guitarist because he played with so much energy.")
calibrationSentences.append("The intern comforted the patient because he was in great pain.")
calibrationSentences.append("The casino hired the daredevil because he was confident that everything would go according to plan.")
calibrationSentences.append("The beggar is often scrounging for cigarettes.")
calibrationSentences.append("The cartoonist who the readers supported pressured the dean because she thought that censorship was never appropriate.")
calibrationSentences.append("The prisoner who the guard attacked tackled the warden although he had no intention of trying to escape.")
calibrationSentences.append("The passer-by threw the cardboard box into the trash-can with great force.")
calibrationSentences.append("The biker who the police arrested ran a light since he was driving under the influence of alcohol.")
calibrationSentences.append("The scientists who were in the lab studied the alien while the blood sample was run through the computer.")
calibrationSentences.append("The student quickly finished his homework assignments.")
calibrationSentences.append("The environmentalist who the demonstrators at the rally supported calmed the crowd until security came and sent everyone home.")
calibrationSentences.append("The producer shoots a new movie every year.")
calibrationSentences.append("The rebels who were in the jungle captured the diplomat after they threatened to kill his family for not complying with their demands.")
calibrationSentences.append("Dinosaurs ate other reptiles during the stone age.")
calibrationSentences.append("The manager who the baker loathed spoke to the new pastry chef because he had instituted a new dress code for all employees.")
calibrationSentences.append("The teacher doubted that the test that had taken him a long time to design would be easy to answer.")
calibrationSentences.append("The cook who the servant in the kitchen hired offended the butler and then left the mansion early to see a movie at the local theater.")





def getTotalSentenceSurprisalsCalibration(SANITY="Sanity", VERBS=2): # Surprisal for EOS after 2 or 3 verbs
    assert SANITY in ["ModelTmp", "Model", "Sanity", "ZeroLoss"]
    assert VERBS in [1,2]
#    print(plain_lm) 
    numberOfSamples = 6
    import scoreWithGPT2Medium as scoreWithGPT2
    with torch.no_grad():
     with open("<<OUTPUT_PATH>>"+__file__+"_"+str(args.load_from_joint)+"_"+SANITY, "w") as outFile:
      print("\t".join(["Sentence", "Region", "Word", "Surprisal", "SurprisalReweighted", "Repetition"]), file=outFile)
      TRIALS_COUNT = 0
      for sentenceID in range(len(calibrationSentences)):
          print(sentenceID)
          sentence = calibrationSentences[sentenceID].lower().replace(".", "").replace(",", "").replace("n't", " n't").split(" ")
          context = sentence[0]
          remainingInput = sentence[1:]
          regions = range(len(sentence))
          print("INPUT", context, remainingInput)
          assert len(remainingInput) > 0
          for i in range(len(remainingInput)):
            for repetition in range(2):
              numerified = encodeContextCrop(" ".join(remainingInput[:i+1]), "later the nurse suggested they treat the patient with an antibiotic but in the end this did not happen . " + context)
              pointWhereToStart = max(0, args.sequence_length - len(context.split(" ")) - i - 1) # some sentences are too long
              assert pointWhereToStart >= 0, (args.sequence_length, i, len(context.split(" ")))
              assert numerified.size()[0] == args.sequence_length+1, (numerified.size())
     #         print(i, " ########### ", SANITY, VERBS)
    #          print(numerified.size())
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
              result, resultNumeric, fractions, thatProbs, amortizedPosterior = autoencoder.sampleReconstructions(numeric, numeric_noised, None, 2, numberOfBatches=numberOfSamples*24, fillInBefore=pointWhereToStart)
              # get THAT fractions

              resultNumeric = resultNumeric.transpose(0,1).contiguous()


#              print(resultNumeric.size(), numeric_noised.size())
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
    #          print(surprisals_past.size(), surprisals_nextWord.size(), amortizedPosterior.size(), likelihood.size())
   #           print(amortizedPosterior.mean(), likelihood.mean(), surprisals_past.mean(), surprisals_nextWord.mean())
              unnormalizedLogTruePosterior = likelihood - surprisals_past
  #            print(unnormalizedLogTruePosterior)
 #             print(amortizedPosterior.mean())
              assert float(unnormalizedLogTruePosterior.max()) <= 1e-5
              assert float(amortizedPosterior.max()) <= 1e-5
              log_importance_weights = unnormalizedLogTruePosterior - amortizedPosterior
              log_importance_weights_maxima, _ = log_importance_weights.max(dim=1, keepdim=True)
#              assert False, "the importance weights seem wacky"
              print(log_importance_weights[0])
              for j in range(24): # TODO the importance weights seem wacky
                 if j % 3 != 0:
                    continue
                 print(j, "@@", result[j], float(surprisals_past[0, j]), float(surprisals_nextWord[0, j]), float(log_importance_weights[0, j]), float(likelihood[0, j]), float(amortizedPosterior[0, j]))
              print(" ".join([itos_total[int(x)] for x in numeric_noised[:, 0].detach().cpu()]))
#              quit()
#              print(log_importance_weights_maxima)
              log_importance_weighted_probs_unnormalized = torch.log(torch.exp(log_importance_weights - log_importance_weights_maxima - surprisals_nextWord).sum(dim=1)) + log_importance_weights_maxima.squeeze(1)
              log_importance_weights_sum = torch.log(torch.exp(log_importance_weights - log_importance_weights_maxima).sum(dim=1)) + log_importance_weights_maxima.squeeze(1)
              reweightedSurprisals = -(log_importance_weighted_probs_unnormalized - log_importance_weights_sum)
              #print(reweightedSurprisals.size())
              #quit()
#              print(log_importance_weighted_probs_unnormalized.size(), log_importance_weights_maxima.size())
              reweightedSurprisalsMean = reweightedSurprisals.mean()
#              quit()

              surprisalOfNextWord = surprisals_nextWord.exp().mean(dim=1).log().mean()
  #            print("PREFIX + NEXT WORD", " ".join([itos_total[int(x)] for x in numerified[:,0]]), surprisalOfNextWord, reweightedSurprisalsMean)
   #           quit()
              # for printing
              nextWordSurprisal_cpu = surprisals_nextWord.view(-1).detach().cpu()
#              reweightedSurprisal_cpu = reweightedSurprisals.detach().cpu()
#              print(nextWordSurprisal_cpu.size())


              for q in range(0, min(3*24, resultNumeric.size()[1]),  24):
                  print("DENOISED PREFIX + NEXT WORD", " ".join([itos_total[int(x)] for x in resultNumeric[:,q]]), float(nextWordSurprisal_cpu[q])) #, float(reweightedSurprisal_cpu[q//24]))
              print("SURPRISAL", i, regions[i], remainingInput[i],float( surprisalOfNextWord), float(reweightedSurprisalsMean))
              print("\t".join([str(w) for w in [sentenceID, regions[i], remainingInput[i], round(float( surprisalOfNextWord),3), round(float( reweightedSurprisalsMean),3), repetition]]), file=outFile)


getTotalSentenceSurprisalsCalibration(SANITY="Model")

