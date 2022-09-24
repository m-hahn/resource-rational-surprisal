import random
import sys

objectiveName = "LM"

model = "REAL_REAL" 

posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 

posFine = set() #[ "``", ",", ":", ".", "''", "$", "ADD", "AFX", "CC",  "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR",  "JJS", "-LRB-", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS",  "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "-RRB-", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  "WDT", "WP", "WP$", "WRB", "XX" ]



from math import log, exp
from random import random, shuffle, randint


from corpusIterator_PTB_Deps import CorpusIterator_PTB

originalDistanceWeights = {}

morphKeyValuePairs = set()

vocab_lemmas = {}

import nltk.tree

corpus_cached = {}
corpus_cached["train"] = CorpusIterator_PTB("PTB", "train")
corpus_cached["dev"] = CorpusIterator_PTB("PTB", "dev")


def descendTree(tree, vocab, posFine, depsVocab):
   label = tree.label()
   for child in tree:
      if type(child) == nltk.tree.Tree:
   #     print((label, child.label()), type(tree))
        key = (label, child.label())
        depsVocab.add(key)
        descendTree(child, vocab, posFine, depsVocab)
      else:
        posFine.add(label)
        word = child.lower()
        if "*-" in word:
           continue
        vocab[word] = vocab.get(word, 0) + 1
    #    print(child)
def initializeOrderTable():
   orderTable = {}
   vocab = {}
   distanceSum = {}
   distanceCounts = {}
   depsVocab = set()
   for partition in ["train", "dev"]:
     for sentenceAndTree in corpus_cached[partition].iterator():
      _, sentence = sentenceAndTree
      #descendTree(sentence, vocab, posFine, depsVocab)

      for line in sentence:
          vocab[line["word"]] = vocab.get(line["word"], 0) + 1
          posFine.add(line["posFine"])
          depsVocab.add(line["dep"])
   return vocab, depsVocab

#import torch.distributions
import torch.nn as nn
import torch
from torch.autograd import Variable

import numpy.random

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()


totalCountRCs = 0
totalCountHeadIsLast = 0


headNouns = {}

def orderSentenceRec(tree, sentence, printThings, linearized):
   global totalCountRCs 
   global totalCountHeadIsLast 

   label = tree.label()
   if label[-1] in "1234567890":
        label = label[:label.rfind("-")]
   children = [child for child in tree]
   if type(children[0]) != nltk.tree.Tree:
      assert all([type(x) != nltk.tree.Tree for x in children])
      assert len(list(children)) == 1, list(children)
      for c in children:
        if label in ["'", ":", "``", ",", "''", "#", ".", "-NONE-"] or label[0] == "-" or "*-" in c:
           continue
        word = sentence[tree.start]["word"] #c.lower(), )
        if word != c.lower().replace("\/","/"):
           print(142, word, c.lower())
        return {"word" : word, "category" : label, "children" : None, "dependency" : "NONE"}
   else:
      assert all([type(x) == nltk.tree.Tree for x in children])
      children = [child for child in children if child.start < child.end] # remove children that consist of gaps or otherwise eliminated tokens

      # find which children seem to be dependents of which other children
      if True or model != "REAL_REAL": 
        childDeps = [None for _ in children]
        childHeads = [None for _ in children]
        for i in range(len(children)):
           incomingFromOutside = [x for x in tree.incoming if x in children[i].incoming]
           if len(incomingFromOutside) > 0:
              childDeps[i] = sentence[incomingFromOutside[-1][1]]["dep"]
              childHeads[i] = sentence[incomingFromOutside[-1][1]]["head"]

              if len(incomingFromOutside) > 1:
                  print("FROM OUTSIDE", [sentence[incomingFromOutside[x][1]]["dep"] for x in range(len(incomingFromOutside))])
           for j in range(len(children)):
              if i == j:
                 continue
              incomingFromJ = [x for x in children[i].incoming if x in children[j].outgoing]
              if len(incomingFromJ) > 0:
                 if len(incomingFromJ) > 1:
                    duplicateDeps = tuple([sentence[incomingFromJ[x][1]]["dep"] for x in range(len(incomingFromJ))])
                    if not (duplicateDeps == ("obj", "xcomp")):
                       print("INCOMING FROM NEIGHBOR", duplicateDeps)
                 childDeps[i] = sentence[incomingFromJ[-1][1]]["dep"]
                 childHeads[i] = sentence[incomingFromJ[-1][1]]["head"]
        assert None not in childDeps, (childDeps, children)
  
        keys = childDeps
  
        childrenLinearized = children
   
      childrenAsTrees = []
      for child, dependency in zip(children, childDeps):
          childrenAsTrees.append(orderSentenceRec(child, sentence, printThings, linearized))
          if childrenAsTrees[-1] is None: # this will happen for punctuation etc 
              del childrenAsTrees[-1]
          else:
             childrenAsTrees[-1]["dependency"] = dependency
      if label == "SBAR":
        if len(childrenAsTrees) > 1: #  
           if len(childrenAsTrees) == 2 and childrenAsTrees[0]["category"] in ["IN", "WHNP"] and childrenAsTrees[1]["category"] == "S" and tree.leaves()[0] == "that": # Relative clause
             RELATION = childrenAsTrees[1]["dependency"]
#             print(RELATION, childrenAsTrees[0]["dependency"])
             if RELATION in ["acl:relcl", "ccomp"]: # and childrenAsTrees[0]["dependency"] == "mark": # Object Relatives
                 #print(len(childrenAsTrees), childrenAsTrees)
                 leaves = [x for x in tree.leaves() if not (x.startswith("*T*") or x.startswith("*U*"))]
                 dominatedBy = sentence[childHeads[1]-1]
                 if dominatedBy["posUni"].startswith("N"):
                   if RELATION == "acl:relcl":
                      print("WORDS       ", " ".join(leaves))
                      print("CATEGORIES  ", list(zip([x["category"] for x in childrenAsTrees], [x["dependency"] for x in childrenAsTrees])))
                      print("Position in matrix clause", dominatedBy, len(tree.leaves()))
                      print(dominatedBy["dep"])
                   headNoun = dominatedBy["lemma"]
                   if headNoun not in headNouns:
                       headNouns[headNoun] = {"ccomp" : {"nsubj" : 0, "other" : 0}, "acl:relcl" : {"nsubj" : 0, "other" : 0}}
                   relation = "nsubj" if dominatedBy["dep"] == "nsubj" else "other"

                   headNouns[headNoun][RELATION][relation] = headNouns[headNoun][RELATION][relation] + 1
                   #print(headNouns)
#                 if sentence[childHeads[1]-1]["dep"] == "nsubj":
#                      print(childrenAsTrees[1])
#                      print("Embedded verb head", sentence[childHeads[0]-1])
#                      print("Is the last word of RC?", sentence[childHeads[0]-1]["word"] == leaves[-1])
#                      totalCountRCs += 1
#                      totalCountHeadIsLast += (1 if (sentence[childHeads[0]-1]["word"] == leaves[-1]) else 0)
#                      print(totalCountHeadIsLast / float(totalCountRCs), totalCountRCs) # for nsubj: around 0.25 (C: [0.1306099, 0.3816907]), for obj: similar, perhaps lower
#                      # What follows the relative clause?
     
        #   else:
#             print(childrenAsTrees)
         #    print(tree.leaves())
          #   print([x["category"] for x in childrenAsTrees])


      return {"category" : label, "children" : childrenAsTrees, "dependency" : "NONE"}

def numberSpans(tree, start, sentence):
   if type(tree) != nltk.tree.Tree:
      if tree.startswith("*") or tree == "0":
        return start, ([]), ([])
      else:
        #print("CHILDREN", start, sentence[start].get("children", []))
        outgoing = ([(start, x) for x in sentence[start].get("children", [])])
        #if len(sentence[start].get("children", [])) > 0:
           #print("OUTGOING", outgoing)
           #assert len(outgoing) > 0
#        if sentence[start]["head"] == 0:
#             print("ROOT", start)
        return start+1, ([(sentence[start]["head"]-1, start)]), outgoing
   else:
      tree.start = start
      incoming = ([])
      outgoing = ([])
      for child in tree:
        start, incomingC, outgoingC = numberSpans(child, start, sentence)
        incoming +=  incomingC
        outgoing += outgoingC
      tree.end = start
      #print(incoming, outgoing, tree.start, tree.end)
   #   print(tree.start, tree.end, incoming, [(hd,dep) for hd, dep in incoming if hd < tree.start or hd>= tree.end])
      incoming = ([(hd,dep) for hd, dep in incoming if hd < tree.start or hd>= tree.end])
      outgoing = ([(hd,dep) for hd, dep in outgoing if dep < tree.start or dep>= tree.end])

      tree.incoming = incoming
      tree.outgoing = outgoing
      #print(incoming, outgoing)
      return start, incoming, outgoing

import copy

def binarize(tree):
   # tree is a single node, i.e. a dict
   if tree["children"] is None:
       return tree
   else:
#       print(tree)
       if len(tree["children"]) <= 1: # remove unary projections
          result = binarize(tree["children"][0]) #{"category" : tree["category"], "dependency" : tree["dependency"], "children" : children}
          result["category"] = tree["category"]
          return result
       else:
          children = [binarize(x) for x in tree["children"][:]]
          left = children[0]
          for child in children[1:]:
             left = {"category" : tree["category"]+"_BAR", "children" : [left, child], "dependency" : tree["dependency"]}
          return left

def orderSentence(tree, printThings):
   global model
   linearized = []
   tree, sentence = tree
   for i in range(len(sentence)):
      line = sentence[i]
      if line["dep"] == "root":
         continue
      head = line["head"] - 1
      if "children" not in sentence[head]:
        sentence[head]["children"] = []
      sentence[head]["children"].append(i)
   end, incoming, outgoing = numberSpans(tree, 0, sentence)
   assert len(incoming) == 1, incoming
   assert len(outgoing) == 0, outgoing
   if (end != len(sentence)):
      print(tree.leaves())
      print([x["word"] for x in sentence])
   return orderSentenceRec(tree, sentence, printThings, linearized)

vocab, depsVocab = initializeOrderTable()


posFine = list(posFine)
itos_pos_fine = posFine
stoi_pos_fine = dict(zip(posFine, range(len(posFine))))



itos_pure_deps = sorted(list(depsVocab)) 
stoi_pure_deps = dict(zip(itos_pure_deps, range(len(itos_pure_deps))))
   

itos_deps = itos_pure_deps
stoi_deps = stoi_pure_deps

#print itos_deps

#dhWeights = [0.0] * len(itos_deps)
distanceWeights = [0.0] * len(itos_deps)


import os

if model == "RANDOM_MODEL":
  for key in range(len(itos_deps)):
     #dhWeights[key] = random() - 0.5
     distanceWeights[key] = random()
  originalCounter = "NA"
elif model == "REAL" or model == "REAL_REAL":
  originalCounter = "NA"
elif model == "RANDOM_BY_TYPE":
  #dhByType = {}
  distByType = {}
  for dep in itos_pure_deps:
 #   dhByType[dep] = random() - 0.5
    distByType[dep] = random()
  for key in range(len(itos_deps)):
#     dhWeights[key] = dhByType[itos_deps[key]]
     distanceWeights[key] = distByType[itos_deps[key]]
  originalCounter = "NA"

lemmas = list(vocab_lemmas.items())
lemmas = sorted(lemmas, key = lambda x:x[1], reverse=True)

words = list(vocab.items())
words = sorted(words, key = lambda x:x[1], reverse=True)
itos = list(map(lambda x:x[0], words))
stoi = dict(list(zip(itos, range(len(itos)))))

if len(itos) > 6:
   assert stoi[itos[5]] == 5


vocab_size = 10000
vocab_size = min(len(itos),vocab_size)

outVocabSize = len(posFine)+vocab_size+3


itos_total = ["EOS", "EOW", "SOS"] + itos_pos_fine + itos[:vocab_size]
assert len(itos_total) == outVocabSize




initrange = 0.1
crossEntropy = 10.0

import torch.nn.functional



counter = 0


lastDevLoss = None
failedDevRuns = 0
devLosses = [] 


lossModule = nn.NLLLoss()
lossModuleTest = nn.NLLLoss(size_average=False, reduce=False, ignore_index=2)

corpusBase = corpus_cached["train"]
corpus = corpusBase.iterator()



# get the initial grammar

# perform splits on the grammar

# run EM

unary_rules = {}

binary_rules = {}

terminals = {}

def addCounts(tree):
   if tree["children"] is None:
      nonterminal = tree["category"]#+"@"+tree["dependency"]       
      terminal = tree["word"]
      if nonterminal not in terminals:
        terminals[nonterminal] = {}
      if terminal not in terminals[nonterminal]:
          terminals[nonterminal][terminal] = 0
      terminals[nonterminal][terminal] += 1
   else:
      for child in tree["children"]:
         addCounts(child)
      if len(tree["children"]) == 1:
         assert False
         nonterminal = tree["category"]#+"@"+tree["dependency"]       
         child = tree["children"][0]
         nonterminalChild = child["category"]#+"@"+child["dependency"]       
         if nonterminal not in unary_rules:
             unary_rules[nonterminal] = {}
         if nonterminalChild not in unary_rules[nonterminal]:
            unary_rules[nonterminal][nonterminalChild] = 0
         unary_rules[nonterminal][nonterminalChild] += 1
      elif len(tree["children"]) == 2:
         nonterminal = tree["category"]#+"@"+tree["dependency"]       
         left  = tree["children"][0]
         right = tree["children"][1]
   
         nonterminalLeft  = left["category"]#+"@"+left["dependency"]       
         nonterminalRight = right["category"]#+"@"+right["dependency"]       
         if nonterminal not in binary_rules:
              binary_rules[nonterminal] = {}
         if (nonterminalLeft, nonterminalRight) not in binary_rules[nonterminal]:
            binary_rules[nonterminal][(nonterminalLeft, nonterminalRight)] = 0
         binary_rules[nonterminal][(nonterminalLeft, nonterminalRight)] += 1


roots = {}


inStackDistribution = {() : 0}

# stack = incomplete constituents that have been started
def updateInStackDistribution(tree, stack):
   if tree["children"] is None:
      return
   else:
     updateInStackDistribution(tree["children"][0], stack + (tree["category"],))
     inStackDistribution[stack] = inStackDistribution.get(stack, 0) + 1
     updateInStackDistribution(tree["children"][1], stack + (tree["category"],))

sentCount = 0
for sentence in corpus:
   sentCount += 1
   ordered = orderSentence(sentence,  sentCount % 50 == 0)



# construct count matrices

# construct grammar

# create split

# run EM

# merge symbols

with open("results_withPosition.tsv", "w") as outFile:
   print("\t".join(["Noun", "Count_CCOMP_nsubj", "Count_CCOMP_other", "Count_Relcl_nsubj", "Count_Relcl_other"]), file=outFile)
   for noun, counts in headNouns.items():
      count_ccomp_nsubj = counts["ccomp"]["nsubj"]
      count_ccomp_other = counts["ccomp"]["other"]
      count_relcl_nsubj = counts["acl:relcl"]["nsubj"]
      count_relcl_other = counts["acl:relcl"]["other"]
      if count_ccomp_other + count_ccomp_nsubj == 0:
         continue
      print("\t".join([noun, str(count_ccomp_nsubj), str(count_ccomp_other), str(count_relcl_nsubj), str(count_relcl_other)]), file=outFile)

