import random
import subprocess
scripts = []

import sys

script = "char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_TPLf.py"

import glob
models = glob.glob("/u/scr/mhahn/CODEBOOKS_MEMORY/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_TPS.py_*.model")
random.shuffle(models)
if len(sys.argv) > 1:
   limit = int(sys.argv[1])
else:
   limit = 1000
count = 0
for model in models:
   ID = model[model.rfind("_")+1:model.rfind(".")]
   resultsPath = f"/u/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/{script}_{ID}_Model"
   if len(glob.glob(resultsPath))>0:
     if os.path.getsize(resultsPath) > 0 and time.time() - os.stat(resultsPath).st_mtime < 3600: # written to within the last hour
       print("WORKING ON THIS?", ID)
       continue
     if os.path.getsize(resultsPath) > 1000:
        print("EXISTS", ID, os.path.getsize(resultsPath))
        continue
     print("TODO", ID, os.path.getsize(resultsPath))
   with open(glob.glob(f"/u/scr/mhahn/reinforce-logs-both-short/results/*_{ID}")[0], "r") as inFile:
      args = dict([x.split("=") for x in next(inFile).strip().replace("Namespace(", "").rstrip(")").split(", ") ])
      delta = float(args["deletion_rate"])
      lambda_ = float(args["predictability_weight"])
      if lambda_ != 1:
        print("FOR NOW DON'T CONSIDER")
        continue
   print("DOES NOT EXIST", ID)
   command = ["/u/nlp/anaconda/main/anaconda3/envs/py36-mhahn/bin/python", script, "--load_from_joint="+ID]
   print(command)
   subprocess.call(command)
   count += 1
   if count >= limit:
     break
