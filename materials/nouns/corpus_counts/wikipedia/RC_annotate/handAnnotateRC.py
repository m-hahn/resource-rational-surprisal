import stanza

from collections import defaultdict                                                                                                                                                                         
import stanza                                                                                                                                                                                               
queue = []                                                                                                                                                                                                  
noun, cont = None, None                                                                                                                                                                                     
useNext = True                                                                                                                                                                                              
import sys                                                                                                                                                                                                  
noun = sys.argv[1]


overallData = []
with open("/u/scr/mhahn/nouns-that-samples/"+noun+"_sent", "r") as inFile:
   for line in inFile:
         if len(line) < 3:
             continue
         label, paragraph = line.strip().split("\t")
         overallData.append([label, paragraph])

LABELS = ["SC", "RC", "Other"]

import os
if not os.path.isfile("/u/scr/mhahn/nouns-that-samples/"+noun+"_lab"):
   with open("/u/scr/mhahn/nouns-that-samples/"+noun+"_lab", "a") as outFile:
      print("", file=outFile)

with open("/u/scr/mhahn/nouns-that-samples/"+noun+"_lab", "r") as inFile:
    for line in inFile:
         if len(line) < 3:
               continue
         index, hash_, label = line.strip().split("\t")
         index = int(index)
         assert overallData[index][1].startswith(hash_), hash_
         assert label in LABELS, (label,)
         overallData[index][0] = label

import statsmodels.stats.proportion
import math


nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', use_gpu=True)                                                                                                                         
for sentIndex in range(len(overallData)):
  if overallData[sentIndex][0] == "_":
    SC = len([None for x in overallData if x[0] == "SC"])
    RC = len([None for x in overallData if x[0] == "RC"])
    print("Counts ", SC, RC)
    confint = statsmodels.stats.proportion.proportion_confint(count=SC+1, nobs=SC+RC+2)
    print(confint[0], confint[1])     
    print("CI: ", math.log(confint[0]+1e-10), math.log(confint[1]))
    print("CI: ", math.log(1-confint[1]+1e-10), math.log(1-confint[0]))
    sentence = overallData[sentIndex][1]
    doc = nlp((sentence))
    for sent in doc.sentences:
      if f"the {noun} that" in sent.text:
        THAT = None
        HEAD = None
        NOUN = None
        out = ""
        for i in range(len(sent.words)):
           if i<len(sent.words)-2 and sent.words[i].text == "the" and sent.words[i+1].text == noun and sent.words[i+2].text == "that":
               NOUN = sent.words[i+1]
               THAT = sent.words[i+2]
               HEAD = sent.words[THAT.head-1]
           if sent.words[i] == NOUN:
             out += '\033[92m'
           elif sent.words[i] == THAT:
             out += '\033[96m'
           elif sent.words[i] == HEAD:
             out += '\033[93m'
           out += sent.words[i].text
           #if sent.words[i] in [HEAD]: # THAT, 
           #  out += "("+sent.words[i].deprel+")"
           if sent.words[i] in [NOUN, THAT, HEAD]:
             out += '\033[0m'
           out += " "
        while True:
          print(sent.text)
          print("")
          print("")
          print("")
          print(out)
          print("   ".join(["("+str(i+1)+") "+LABELS[i] for i in range(len(LABELS))]))
          reply = sys.stdin.readline()
          try:
             reply = int(reply)
          except ValueError:
             reply = reply.strip()
             pass
          if reply == 0:
             break
          elif reply in [1,2,3]:
             with open("/u/scr/mhahn/nouns-that-samples/"+noun+"_lab", "a") as outFile:
                 print("\t".join([str(sentIndex), sentence[:10].strip(), LABELS[reply-1]]), file=outFile)
             overallData[sentIndex][0] = LABELS[reply-1]
             break
        break    

#        print(THAT.deprel, HEAD.text, HEAD.deprel)
       
#               print("_\t"+sent.text, file=outFile)
#     #       quit()
         
