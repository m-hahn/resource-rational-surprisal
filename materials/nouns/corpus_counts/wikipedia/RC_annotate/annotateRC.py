import stanza

from collections import defaultdict                                                                                                                                                                         
import stanza                                                                                                                                                                                               
nlp = stanza.Pipeline('en', processors='tokenize', use_gpu=True)                                                                                                                         
queue = []                                                                                                                                                                                                  
noun, cont = None, None                                                                                                                                                                                     
useNext = True                                                                                                                                                                                              
import sys                                                                                                                                                                                                  
import os


strings = []

for nounFile in os.listdir("/u/scr/mhahn/nouns-that-samples/"):
  noun = nounFile[nounFile.rfind("/")+1:]
  if "_" in noun:
     continue
  if noun.startswith("."):
     continue
  if os.path.isfile("/u/scr/mhahn/nouns-that-samples/"+noun+"_sent"):
      continue
  print(noun)
  overallData = []
  with open("/u/scr/mhahn/nouns-that-samples/"+noun, "r") as inFile:
   with open("/u/scr/mhahn/nouns-that-samples/"+noun+"_sent", "w") as outFile:
     for line in inFile:
         if len(line) < 4:
             continue
         label, paragraph = line.strip().split("\t")
         overallData.append(paragraph)
  
         doc = nlp((paragraph))
         
         for sent in doc.sentences:
            if f"the {noun} that" in sent.text:
               print("_\t"+sent.text, file=outFile)
     #       quit()
         
