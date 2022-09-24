import os
import sys
import random
from collections import defaultdict
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="english")
parser.add_argument("--load-from-lm", dest="load_from_lm", type=str, default=964163553) # language model taking noised input # Amortized Prediction Posterior
parser.add_argument("--load-from-autoencoder", dest="load_from_autoencoder", type=str, default=random.choice([647336050, 516252642, 709961927, 727001672, 712478284, 524811876])) # Amortized Reconstruction Posterior
parser.add_argument("--load-from-plain-lm", dest="load_from_plain_lm", type=str, default=random.choice([27553360, 935649231])) # plain language model without noise (Prior)


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
parser.add_argument("--learning_rate_lm", type = float, default= random.choice([0.0002, 0.0005, 0.001])) # 0.0001, 
parser.add_argument("--lr_decay", type=float, default=random.choice([1.0]))
parser.add_argument("--reward_multiplier_baseline", type=float, default=0.1)
parser.add_argument("--dual_learning_rate", type=float, default=random.choice([0.01, 0.02, 0.05, 0.1, 0.2, 0.3]))
parser.add_argument("--momentum", type=float, default=random.choice([0.5, 0.7, 0.7, 0.7, 0.7, 0.9])) # Momentum is helpful in facilitating convergence to a low-loss solution (March 25, 2021). It might be even more important for getting fast convergence than a high learning rate
parser.add_argument("--entropy_weight", type=float, default=random.choice([0.0])) # 0.0,  0.005, 0.01, 0.1, 0.4]))



# Control
parser.add_argument("--clip_lm", type=bool, default=random.choice([False, True, True, True]))
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--tuning", type=int, default=1) #random.choice([0.00001, 0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.0008, 0.001])) # 0.0,  0.005, 0.01, 0.1, 0.4]))

# Lambda and Delta Parameters
parser.add_argument("--deletion_rate", type=float, default=0.5)
parser.add_argument("--predictability_weight", type=float, default=1)


TRAIN_LM = True
assert TRAIN_LM



model = "REAL_REAL"

import math

args=parser.parse_args()

############################


assert args.learning_rate_lm < 0.01

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
     # Word embedding matrix (will be initialized from the prediction posterior)
     self.word_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=2*args.word_embedding_size).cuda()
     # Modules of the memory model
     self.modules_memory = [self.memory_mlp_inner, self.memory_mlp_outer, self.memory_mlp_inner_from_pos, self.positional_embeddings, self.perword_baseline_inner, self.perword_baseline_outer, self.memory_word_pos_inter, self.memory_bilinear, self.memory_mlp_inner_bilinear, self.word_embeddings]

  def forward(self, numeric):
      embedded_everything_mem = self.word_embeddings(numeric)

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


# Dual parameter (for Lagrangian dual)
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


assert TRAIN_LM
optim_autoencoder = torch.optim.SGD(parameters_autoencoder(), lr=args.learning_rate_autoencoder, momentum=0.0) # 0.02, 0.9
optim_memory = torch.optim.SGD(parameters_memory(), lr=args.learning_rate_memory, momentum=args.momentum) # 0.02, 0.9
if args.predictability_weight > 0:
   optim_lm = torch.optim.SGD(parameters_lm(), lr=args.learning_rate_lm, momentum=0.0) # 0.02, 0.9

###############################################3


# Load pretrained prior and amortized posteriors

# Amortized Reconstruction Posterior
if args.load_from_autoencoder is not None:
  print(args.load_from_autoencoder)
  checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+"autoencoder2_mlp_bidir_Erasure_SelectiveLoss.py"+"_code_"+str(args.load_from_autoencoder)+".txt")
  for i in range(len(checkpoint["components"])):
      autoencoder.modules_autoencoder[i].load_state_dict(checkpoint["components"][i])
  del checkpoint
 
# Amortized Prediction Posterior
if args.load_from_lm is not None:
  lm_file = "char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_NoChars_Erasure.py"
  checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+lm_file+"_code_"+str(args.load_from_lm)+".txt")
  for i in range(len(checkpoint["components"])):
      lm.modules_lm[i].load_state_dict(checkpoint["components"][i])
  del checkpoint

from torch.autograd import Variable

# Initialize memory word embeddings from LM
memory.word_embeddings.weight.data.copy_(lm.word_embeddings.weight.data)


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

      # Step 3: Compute control variate
      baselineValues = 10*memory.sigmoid(memory.perword_baseline_outer(memory.relu(memory.perword_baseline_inner(embedded_everything_mem[-1].detach())))).squeeze(1)
      assert tuple(baselineValues.size()) == (NUMBER_OF_REPLICATES,)


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

      input_tensor_pure = Variable(numeric[:-1], requires_grad=False)
      input_tensor_noised = Variable(numeric_noised[:-1], requires_grad=False)
      target_tensor_full = Variable(numeric[1:], requires_grad=False)

      target_tensor_onlyNoised = Variable(numeric_onlyNoisedOnes[1:], requires_grad=False)
      #####################################################################################


      ##########################################
      ##########################################
      # Step 5: Run reconstruction inference network
      autoencoder_lossTensor =  autoencoder.forward(input_tensor_pure, input_tensor_noised, target_tensor_onlyNoised, NUMBER_OF_REPLICATES)

      ##########################################
      ##########################################
      # Step 6: Run prediction inference network
      if args.predictability_weight > 0:
       lm_lossTensor = lm.forward(input_tensor_noised, target_tensor_full, NUMBER_OF_REPLICATES)
      ##########################################
      ##########################################

      # Step 7: Collect loss function for training signal
      # Reward, term 1
      if args.predictability_weight > 0:
        negativeRewardsTerm1 = 2*args.predictability_weight * lm_lossTensor.mean(dim=0) + 2*(1-args.predictability_weight) * autoencoder_lossTensor.mean(dim=0)
      else:
        negativeRewardsTerm1 = autoencoder_lossTensor.mean(dim=0)


      # Reward, term 2
      # Regularization towards lower retention rates
      negativeRewardsTerm2 = memory_filter.mean(dim=0)
      retentionTarget = 1-args.deletion_rate
      loss = 0

      # Reconstruction Loss (referred to as L2 in SI Appendix, Section 1)
      loss += autoencoder_lossTensor.mean()

      # Prediction Loss (training signal for pediction inference network)
      if args.predictability_weight > 0:
         loss += lm_lossTensor.mean() 

      # Overall Reward (referred to as L1 in SI Appendix, Section 1)
      negativeRewardsTerm = negativeRewardsTerm1 + dual_weight * (negativeRewardsTerm2-retentionTarget)
      # for the dual weight
      loss += (dual_weight * (negativeRewardsTerm2-retentionTarget).detach()).mean()
      if printHere:
          print(negativeRewardsTerm1.mean(), dual_weight, negativeRewardsTerm2.mean(), retentionTarget)

      # baselineValues: the baselines for the prediction loss (term 1)
      # memory_hidden: baseline for term 2
      # Important to detach all but the baseline values

      # Reward Minus Baseline
      # Detached surprisal and mean retention
#      rewardMinusBaseline = (negativeRewardsTerm.detach() - baselineValues - args.RATE_WEIGHT * memory_hidden.mean(dim=0).squeeze(dim=1).detach())

      # Subtract control variate for unbiased variance reduction
      rewardMinusBaseline = (negativeRewardsTerm.detach() - baselineValues - (dual_weight * (memory_hidden.mean(dim=0).squeeze(dim=1) - retentionTarget)).detach())

      # Apply REINFORCE estimator
      # Important to detach from the baseline!!! 
      loss += (rewardMinusBaseline.detach() * bernoulli_logprob_perBatch.squeeze(1)).mean()
      if args.entropy_weight > 0:
         loss -= args.entropy_weight  * entropy

      # Training signal for control variate
      loss += args.reward_multiplier_baseline * rewardMinusBaseline.pow(2).mean()


      ############################
      # Construct running averages
      factor = 0.9996 ** args.batchSize

      # Update running averages
      global runningAverageBaselineDeviation
      global runningAveragePredictionLoss
      global runningAverageReconstructionLoss
      global runningAverageReward
      global expectedRetentionRate

      expectedRetentionRate = factor * expectedRetentionRate + (1-factor) * float(memory_hidden.mean())
      runningAverageBaselineDeviation = factor * runningAverageBaselineDeviation + (1-factor) * float((rewardMinusBaseline).abs().mean())

      if args.predictability_weight > 0:
       runningAveragePredictionLoss = factor * runningAveragePredictionLoss + (1-factor) * round(float(lm_lossTensor.mean()),3)
      runningAverageReconstructionLoss = factor * runningAverageReconstructionLoss + (1-factor) * round(float(autoencoder_lossTensor.mean()),3)
      runningAverageReward = factor * runningAverageReward + (1-factor) * float(negativeRewardsTerm.mean())
      ############################

      if printHere:
         if args.predictability_weight > 0:
          lm_losses = lm_lossTensor.data.cpu().numpy()
         autoencoder_losses = autoencoder_lossTensor.data.cpu().numpy()

         numericCPU = numeric.cpu().data.numpy()
         numeric_noisedCPU = numeric_noised.cpu().data.numpy()
         memory_hidden_CPU = memory_hidden[:,0,0].cpu().data.numpy()

               
         print("PREDICTION_LOSS", runningAveragePredictionLoss, "RECONSTRUCTION_LOSS", runningAverageReconstructionLoss, "\tTERM2", round(float(negativeRewardsTerm2.mean()),3), "\tAVERAGE_RETENTION", expectedRetentionRate, "\tDEVIATION FROM BASELINE", runningAverageBaselineDeviation, "\tREWARD", runningAverageReward, "\tENTROPY", float(entropy))
         print(dual_weight)
      if updatesCount % 5000 == 0:
         print("updatesCount", updatesCount, updatesCount/maxUpdates)
         print("\t".join([str(x) for x in ("PREDICTION_LOSS", runningAveragePredictionLoss, "RECONSTRUCTION_LOSS", runningAverageReconstructionLoss, "\tTERM2", round(float(negativeRewardsTerm2.mean()),3), "\tAVERAGE_RETENTION", expectedRetentionRate, "\tDEVIATION FROM BASELINE", runningAverageBaselineDeviation, "\tREWARD", runningAverageReward, "\tENTROPY", float(entropy))]), file=sys.stderr)

      #runningAveragePredictionLoss = 0.95 * runningAveragePredictionLoss + (1-0.95) * float(negativeRewardsTerm1.mean())
      
      return loss, product(target_tensor_full.size())


def backward(loss, printHere):
      """ An optimization step for the resource-rational objective function """
      # Set stored gradients to zero
      optim_autoencoder.zero_grad()
      optim_memory.zero_grad()
      if args.predictability_weight > 0:
         optim_lm.zero_grad()

      if dual_weight.grad is not None:
         dual_weight.grad.data.fill_(0.0)
      if printHere:
         print(loss)
      # Calculate new gradients
      loss.backward()
      # Gradient clipping
      torch.nn.utils.clip_grad_value_(parameters_memory_cached, 5.0) #, norm_type="inf")
      if TRAIN_LM and args.clip_lm and args.predictability_weight > 0:
         #assert False
         torch.nn.utils.clip_grad_value_(parameters_lm_cached, 5.0) #, norm_type="inf")

      # Adapt parameters
      optim_autoencoder.step()
      optim_memory.step()
      if args.predictability_weight > 0:
         optim_lm.step()

#      print(dual_weight.grad)
      dual_weight.data.add_(args.dual_learning_rate*dual_weight.grad.data)
 #     print("W", dual_weight)
      dual_weight.data.clamp_(min=0)
  #    print("W", dual_weight)

lossHasBeenBad = 0

import time

totalStartTime = time.time()

lastSaved = (None, None)
devLosses = []
updatesCount = 0

maxUpdates = 200000 if args.tuning == 1 else 10000000000

def showAttention(word, POS=""):
    attention = forward(torch.cuda.LongTensor([stoi[word]+3 for _ in range(args.sequence_length+1)]).view(-1, 1), train=True, printHere=True, provideAttention=True)
    attention = attention[:,0,0]
    print(*(["SCORES", word, "\t"]+[round(x,2) for x in list(attention.cpu().data.numpy())] + (["POS="+POS] if POS != "" else [])))






    

# Helper Functions









def divideDicts(y, z):
   r = {}
   for x in y:
     r[x] = y[x]/z[x]
   return r



startTimePredictions = time.time()



startTimeTotal = time.time()
startTimePredictions = time.time()
startTimeTotal = time.time()

for epoch in range(1000):
   print(epoch)

   # Get training data
   training_data = corpusIteratorWikiWords.training(args.language)
   print("Got data")
   training_chars = prepareDatasetChunks(training_data, train=True)


   # Set the model up for training
   lm.rnn_drop.train(True)
   startTime = time.time()
   trainChars = 0
   counter = 0
   # End optimization when maxUpdates is reached
   if updatesCount > maxUpdates:
     break
   while updatesCount <= maxUpdates:
      counter += 1
      updatesCount += 1
      # Get model predictions at the end of optimization
      if updatesCount == maxUpdates:


          
       # Record reconstructions and surprisals
       with open("/u/scr/mhahn/reinforce-logs-both-short/full-logs/"+__file__+"_"+str(args.myID), "w") as outFile:
         startTimePredictions = time.time()

         sys.stdout = outFile
         print(updatesCount, "Slurm", os.environ["SLURM_JOB_ID"])
         print(args)
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

     # Get a batch from the training set
      try:
         numeric, _ = next(training_chars)
      except StopIteration:
         break
      printHere = (counter % 50 == 0)
      # Run this through the model: forward pass of the resource-rational objective function
      loss, charCounts = forward(numeric, printHere=printHere, train=True)
      # Calculate gradients and update parameters
      backward(loss, printHere)


      # Bad learning rate parameters might make the loss explode. In this case, stop.
      if lossHasBeenBad > 100:
          print("Loss exploding, has been bad for a while")
          print(loss)
          assert False
      trainChars += charCounts 
      if printHere:
          print(("Loss here", loss))
          print((epoch, "Updates", updatesCount, str((100.0*updatesCount)/maxUpdates)+" %", maxUpdates, counter, trainChars, "ETA", ((time.time()-startTimeTotal)/updatesCount * (maxUpdates-updatesCount))/3600.0, "hours"))
          print("Dev losses")
          print(devLosses)
          print("Words per sec "+str(trainChars/(time.time()-startTime)))
          print(args.learning_rate_memory, args.learning_rate_autoencoder)
          print("Slurm", os.environ["SLURM_JOB_ID"])
          print(lastSaved)
          print(__file__)
          print(args)



with open("/u/scr/mhahn/reinforce-logs-both-short/results/"+__file__+"_"+str(args.myID), "w") as outFile:
   print(args, file=outFile)
   print(runningAverageReward, file=outFile)
   print(expectedRetentionRate, file=outFile)
   print(runningAverageBaselineDeviation, file=outFile)
   print(runningAveragePredictionLoss, file=outFile)
   print(runningAverageReconstructionLoss, file=outFile)


state = {"arguments" : args, "words" : itos, "memory" : [c.state_dict() for c in memory.modules_memory], "autoencoder" : [c.state_dict() for c in autoencoder.modules_autoencoder]}
torch.save(state, f"/u/scr/mhahn/CODEBOOKS_MEMORY/{__file__}_{args.myID}.model")



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


