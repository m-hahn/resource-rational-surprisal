PATH = "/u/scr/mhahn/reinforce-logs-both-short/results/"
import os
import sys


from collections import defaultdict

results = defaultdict(list)
for name in os.listdir(PATH):
  if "NoComma_Bugfix" in name:
   with open(PATH+name, "r") as inFile:
      args = next(inFile).strip()
      runningAverageReward = float(next(inFile).strip())
      expectedRetentionRate = float(next(inFile).strip())
      runningAverageBaselineDeviation = float(next(inFile).strip())
      runningAveragePredictionLoss = float(next(inFile).strip())
      runningAverageReconstructionLoss = float(next(inFile).strip())
      args = dict([x.split("=") for x in args.replace("Namespace(", "").rstrip(")").split(", ")])
      deletion_rate = args["deletion_rate"]
      predictability_weight = args["predictability_weight"]
      args["script"] = name[:name.rfind("_")][-10:]
#      print(name, args["script"])
 #     quit()
      results[(predictability_weight, deletion_rate)].append((runningAverageReward, runningAveragePredictionLoss, runningAverageReconstructionLoss, args, expectedRetentionRate))


with open(f"{__file__}.txt", "w") as outFile:
 with open(f"{__file__}.tsv", "w") as outFileTSV:
  for lambda_, delta_ in sorted(list(results)):
    scores = sorted(results[(lambda_, delta_)], key=lambda x:x[0], reverse=True)
    print("-------------------", file=outFile)
    print("-------------------")
    print(lambda_, delta_, file=outFile)
    print(lambda_, delta_)
    for x, a, b, y, z in scores:
       print("\t".join([str(w) for w in [lambda_, delta_, round(z, 2), round(x/2, 2), round(a,2), round(b,2), y["learning_rate_memory"], y["learning_rate_autoencoder"], float(y.get("learning_rate_lm", 0)), y["momentum"], y["myID"], y["script"]]]), file=outFileTSV)
       print("\t".join([str(w) for w in [round(x/2, 2), round(a,2), round(b,2), y["learning_rate_memory"], y["learning_rate_autoencoder"], float(y.get("learning_rate_lm", 0)), y["momentum"], y["myID"]]]), file=outFile)
       print("\t".join([str(w) for w in [round(x/2, 2), round(a,2), round(b,2), y["learning_rate_memory"], y["learning_rate_autoencoder"], float(y.get("learning_rate_lm", 0)), y["momentum"], y["myID"]]]))
    

