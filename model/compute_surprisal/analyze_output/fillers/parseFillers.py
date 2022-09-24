import stanza
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_pretokenized=True)


fillers = set()
with open("../../../../experiments/maze/experiment2/Submiterator-master/trials-experiment2.tsv", "r") as inFile:
 with open(f"output/{__file__}.tsv", "w") as outFile:
   header = next(inFile)
   for line in inFile:
     line = line.strip().split("\t") 
     if line[3].startswith("Filler_"):
        sentence = line[10].replace(",", "").replace(".", " .")
        if sentence not in fillers:
           print(sentence)
           fillers.add(sentence)     
           doc = nlp(sentence)
           sentence = [word for sent in doc.sentences for word in sent.words]
           for i in range(len(sentence)):
              intervening = "NA"
              if sentence[i].pos in ["VERB", "NOUN", "PROPN"]:
                intervening = 0
                for j in range(i):
                  if sentence[j].pos not in ["VERB", "NOUN", "PROPN"]:
                    continue
                  if i+1 == sentence[j].head or j+1 == sentence[i].head:
                    intervening += len([k for k in range(j+1,i) if sentence[k].pos in ["VERB", "NOUN", "PROPN"]])
#                    print(intervening)
              word = sentence[i]
              print(f'{line[3]}\t{word.id}\t{word.text}\t{word.head}\t{word.pos}\t{word.deprel}\t{intervening}', file=outFile)
#           print(*[f'{line[3]}\t{word.id}\t{word.text}\t{word.head}\t{word.pos}\t{word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')
           
           

