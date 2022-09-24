# from https://www.asc.ohio-state.edu/demarneffe.1/LING5050/material/structured.html
header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]



from nltk.corpus import ptb
import os

def addTrees(sec, trees):
   secNum = ("" if sec >= 10 else "0") + str(sec)

   files = os.listdir("/u/scr/corpora/ldc/1999/LDC99T42-treebank_3/parsed/mrg/wsj/"+secNum)
   for name in files:
      for tree in ptb.parsed_sents("WSJ/"+secNum+"/"+name):
         leaves = " ".join([("(" if x == "-LRB-" else (")" if x == "-RRB-" else x.replace("\/", "/").replace("\*","*"))) for x in tree.leaves() if "*-" not in x and (not x.startswith("*")) and x not in ["0", "*U*", "*?*"]])
         if leaves not in deps: # only applies to one sentence in the training partition
            print(leaves)
            continue
         trees.append((tree, deps[leaves]))
          

def getPTB(partition):
   trees = []
   if partition == "train":
     sections = range(0, 19) #19) # 19
   elif partition in ["dev", "valid"]: # 19-21
     sections = range(19, 22) # 22
   elif partition == "test": # 22-24
     sections = range(22, 25)
   sections = [1]
   for sec in sections:
      print(sec)
      addTrees(sec, trees)
   return trees

#print(getPTB("train"))
import os
import random
import sys



with open("/u/scr/mhahn/CORPORA/ptb-ud2/ptb-ud2.conllu", "r") as inFile:
   deps = inFile.read().strip().split("\n\n")
for i in range(len(deps)):
    words = " ".join([x.split("\t")[1] for x in deps[i].split("\n")   ])
    deps[i] = (words, deps[i])
deps = dict(deps)
print(len(deps))
print("Done reading deps")

class CorpusIterator_PTB():
   def __init__(self, language, partition="train"):
      data = getPTB(partition)
#      if shuffleData:
#       if shuffleDataSeed is None:
#         random.shuffle(data)
#       else:
#         random.Random(shuffleDataSeed).shuffle(data)

      self.data = data
      self.partition = partition
      self.language = language
      assert len(data) > 0, (language, partition)
   def permute(self):
      random.shuffle(self.data)
   def length(self):
      return len(self.data)
   def getSentence(self, index):
      result = self.processSentence(self.data[index])
      return result
   def iterator(self):
     for sentence in self.data:
        yield self.processSentence(sentence)
   def processSentence(self, sentenceAndTree):
        tree, sentence = sentenceAndTree
        sentence = list(map(lambda x:x.split("\t"), sentence.split("\n")))
        result = []
        for i in range(len(sentence)):
#           print sentence[i]
           if sentence[i][0].startswith("#"):
              continue
           if "-" in sentence[i][0]: # if it is NUM-NUM
              continue
           if "." in sentence[i][0]:
              continue
           sentence[i] = dict([(y, sentence[i][x]) for x, y in enumerate(header)])
           sentence[i]["head"] = int(sentence[i]["head"])
           sentence[i]["index"] = int(sentence[i]["index"])
           sentence[i]["word"] = sentence[i]["word"].lower()
           if self.language == "Thai-Adap":
              assert sentence[i]["lemma"] == "_"
              sentence[i]["lemma"] = sentence[i]["word"]
           if "ISWOC" in self.language or "TOROT" in self.language:
              if sentence[i]["head"] == 0:
                  sentence[i]["dep"] = "root"

#           if self.splitLemmas:
 #             sentence[i]["lemmas"] = sentence[i]["lemma"].split("+")

  #         if self.storeMorph:
   #           sentence[i]["morph"] = sentence[i]["morph"].split("|")

    #       if self.splitWords:
     #         sentence[i]["words"] = sentence[i]["word"].split("_")


           sentence[i]["dep"] = sentence[i]["dep"].lower()

           result.append(sentence[i])
 #          print sentence[i]
        return (tree,result)



