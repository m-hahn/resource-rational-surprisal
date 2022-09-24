import glob
files = sorted(glob.glob("/u/scr/mhahn/reinforce-logs-both-short/full-logs/char*GPT2M_*S.py_*"))
with open(f"raw_output/{__file__}.tsv", "w") as outFile:
 print("\t".join(["Word", "Distance", "RetentionProb", "POS", "deletion_rate", "predictability_weight", "ID", "Script"]), file=outFile)
 for f in files:
   print(f)
   with open(f, "r") as inFile:
     for line in inFile:
       if line.startswith("Namespace"):
          args = dict([x.split("=") for x in line.replace("Namespace(", "").rstrip(")").split(", ")])
       if line.startswith("SCORES"): # and "POS=" in line:
         line = line.split("\t")
         word = line[0].replace("SCORES", "").strip()
         probs = line[1].strip().split(" ")
         if "POS" in probs[-1]:
            POS = probs[-1].replace("POS=", "")
            probs = probs[:-1]
         else:
            POS = "NA"
         probs = [float(x) for x in probs]
         suffix = f[:f.rfind("_")-3]
         suffix = suffix[-7:]
         for i in range(len(probs)-1, -1, -1):
            print("\t".join([str(q) for q in [word, len(probs)-i, probs[i], POS, args["deletion_rate"], args["predictability_weight"], args["myID"], suffix]]), file=outFile)
     
