import sys
import os

PATH = "/u/scr/mhahn/reinforce-logs/results/"

logs = os.listdir(PATH)

for lossType in ["Deletion", "Erasure"]:
   print("###############")
   results = []
   for filen in logs:
      if lossType not in filen:
         continue
      data = open(PATH+filen, "r").read().strip().split("\n")
      if len(data) == 2:
         continue
         data.append("-1")
      if len(data) == 1:
         continue
      params, perform, memRate,  = data
      params = params.replace("Namespace(", "")[:-1].split(", ")
      load_from_autoencoder = [x.split("=")[1] for x in params if x.startswith("load_from_autoencoder")][0]
      params = [x for x in params if x.split("=")[0] in ["RATE_WEIGHT", "batchSize", "entropy_weight", "learning_rate", "momentum"]]
      params = [x.replace("learning", "learn").replace("entropy", "ent").replace("momentum", "mom").replace("batchSize", "batch") for x in params]

      memRate = memRate.replace("tensor(", "").replace(", device='cuda:0', grad_fn=<MeanBackward0>)", "")
      rate = float(params[0].split("=")[1])
      performance = round(float(perform),4)
      memRate = round(float(memRate),4)
      if memRate < 0.15 and "autoencoder2_mlp_bidir_Deletion_Reinforce2_Tuning_Long_Both_Saving.py" in filen:
            performance = 10
      results.append((rate, performance, memRate, " ".join(params), filen, load_from_autoencoder))
   results = sorted(results, reverse=True)
   lastR = None
   for r in results:
      if lastR is not None and lastR[0] != r[0]:
         print("-----------")
      print("\t".join([str(x) for x in r]))
      lastR = r
