import random
import subprocess
scripts = []

import sys
if len(sys.argv) > 1:
  stimulus_file = sys.argv[1]
else:
  stimulus_file = "Staub_2016" #random.choice(["BartekEtal", "Staub2006", "Staub_2016"])

if stimulus_file == "BartekEtal":
   criticalRegions="Critical_0"
elif stimulus_file == "BartekGG":
   criticalRegions="Critical_0"
elif stimulus_file == "Staub2006":
   criticalRegions = "NP1_0,NP1_1,OR,NP2_0,NP2_1"
elif stimulus_file == "cunnings-sturt-2018":
   criticalRegions = "critical"
elif stimulus_file == "Staub_2016":
   criticalRegions = "V0,D1,N1,V1"
elif stimulus_file == "V11_E1_EN":
   criticalRegions = "Critical_0"
elif stimulus_file == "TaborHutchins":
   criticalRegions = "Final_0"
elif stimulus_file == "Chen2005":
   criticalRegions = "critical_1,critical_2,critical_3,critical_4,critical_5"
elif stimulus_file == "VanDyke_Lewis_2003":
   criticalRegions = "R4_0,R4_1"
else:
   assert False, stimulus_file

script = "char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_Lo.py"

import glob
models = glob.glob("/u/scr/mhahn/CODEBOOKS_MEMORY/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_S.py_*.model")
random.shuffle(models)
limit = 1000
count = 0
for model in models:
   ID = model[model.rfind("_")+1:model.rfind(".")]
   if len(glob.glob(f"/u/scr/mhahn/reinforce-logs-both-short/stimuli-full-logs-tsv/{script}_{stimulus_file}_{ID}_Model"))>0:
     print("EXISTS", ID)
     continue
   with open(glob.glob(f"/u/scr/mhahn/reinforce-logs-both-short/results/*_{ID}")[0], "r") as inFile:
      args = dict([x.split("=") for x in next(inFile).strip().replace("Namespace(", "").rstrip(")").split(", ") ])
      delta = float(args["deletion_rate"])
      lambda_ = float(args["predictability_weight"])
#      if lambda_ != 1:
 #       print("FOR NOW DON'T CONSIDER")
#      if delta < 0.35 or delta > 0.6:
 #       print("FOR NOW DON'T CONSIDER", ID, delta)
#        continue
   print("DOES NOT EXIST", ID, delta, lambda_)
   command = ["/u/nlp/anaconda/main/anaconda3/envs/py36-mhahn/bin/python", script, "--stimulus_file="+stimulus_file, "--criticalRegions="+criticalRegions, "--load_from_joint="+ID]
   print(command)
   subprocess.call(command)
   count += 1
   if count >= limit:
     break
