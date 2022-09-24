import glob
import os
#for deletion_rate in [x/20 for x in range(21)]:
#  for predictability_weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
#     os.mkdir(f"/u/scr/mhahn/ZENODO_RESRAT/pred_{predictability_weight}_del_{deletion_rate}_simple/")
#     os.mkdir(f"/u/scr/mhahn/ZENODO_RESRAT/pred_{predictability_weight}_del_{deletion_rate}_full/")
#quit()  


with open("char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_S.py.tsv", "r") as inFile:
 datas = [x.split("\t") for x in inFile.read().strip().split("\n")]
 for data in datas:
  config = dict([x.split("=") for x in data[1][10:-1].split(", ")])
  ID = data[0]
  path = glob.glob(f"/juice/scr/mhahn/CODEBOOKS_MEMORY/*_{ID}.model")
  if len(path) < 1:
     print("FILE MISSING !!!", config, config["deletion_rate"], config["predictability_weight"])
     continue
  assert len(path) == 1, path
  filename = path[0].split("/")[-1]
  predictability_weight = config["predictability_weight"]
  deletion_rate = config["deletion_rate"]
  #os.symlink("TMP", "TMP2")
  print(f"/u/scr/mhahn/ZENODO_RESRAT/pred_{predictability_weight}_del_{deletion_rate}_simple/{filename}")
  os.symlink(path[0], f"/u/scr/mhahn/ZENODO_RESRAT/pred_{predictability_weight}_del_{deletion_rate}_simple/{filename}")
  #quit()
