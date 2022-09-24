import sys
import random
import subprocess
scripts = []

import sys

script = "char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_TPS.py"

from collections import defaultdict

import glob
for _ in range(int(sys.argv[1])):


   countsByConfig = defaultdict(int)
   configurations = set()
   for i in range(35, 60, 5):
#     if i/100 < 0.2 or i/100 > 0.75:
 #      continue
     for j in [0.75]: #[0, 0.25, 0.5, 0.75, 1]:
        configurations.add((i/100,j))

   logs = glob.glob(f"/u/scr/mhahn/reinforce-logs-both-short/results/{script}_*")

   for log in logs:
      with open(log, "r") as inFile:
          args = dict([x.split("=") for x in next(inFile).strip().replace("Namespace(", "").rstrip(")").split(", ")])
      try:
         countsByConfig[(float(args["deletion_rate"]), float(args["predictability_weight"]))] += 1
         if float(args["deletion_rate"]) >= 0.2 and float(args["deletion_rate"]) < 0.8:
          if countsByConfig[float(args["deletion_rate"]), float(args["predictability_weight"])] >= 5:
            configurations.remove((float(args["deletion_rate"]), float(args["predictability_weight"])))
         else:
          if countsByConfig[float(args["deletion_rate"]), float(args["predictability_weight"])] >= 5:
            configurations.remove((float(args["deletion_rate"]), float(args["predictability_weight"])))
      except KeyError:
         pass
      print(configurations)
   if len(configurations) == 0:
     break
   deletion_rate, predictability_weight = random.choice(list(configurations) )
   command = ["/u/nlp/anaconda/main/anaconda3/envs/py36-mhahn/bin/python", script, f"--deletion_rate={deletion_rate}", f"--predictability_weight={predictability_weight}"]
   print(command)
   subprocess.call(command)

