import glob
import codecs
import os

header0 = "Sentence Region Word Surprisal SurprisalReweighted Copy".split(" ")
header =  header0 + ["Script", "ID", "predictability_weight", "deletion_rate", "autoencoder", "lm"]

PATH0 = "/juice/scr/mhahn/reinforce-logs-both-short/results/"
PATH = "/juice/scr/mhahn/reinforce-logs-both-short/full-logs/"
PATH2 = "/juice/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/"

with open(f"{PATH2}/{__file__}.tsv", "w") as outFile:
 print("\t".join(header), file=outFile)
 for f in sorted(os.listdir(PATH2)):
   shib = "12_NormJudg_Short_Cond_Shift_NoComma_Bugfix"
   if shib in f:
      suffix = "script_"+f[f.index(shib)+len(shib):f.index(".py")]
      if "_VN3Stims_" not in f or "_OnlyLoc" in f or "ZERO" in f or f.endswith("ModelTmp") or "EYE" in f or "Lf" not in f:
        continue
 #     print(f)
      #print(suffix)
      assert f.endswith("_Model"), f
      ID = f.replace("_Model", "").split("_")[-1]
      logPath = PATH0+"/*_"+ID
      relevantFile = glob.glob(logPath)
      if len(relevantFile) == 0:
        print("ERROR NO LOG FOUND", f)
        continue
      with codecs.open(relevantFile[0], "r", 'utf-8', "ignore") as inFile:
         try:
           arguments = next(inFile).strip()
         except StopIteration:
           print("CANNOT FIND ARGUMENTS", f)
           continue
#         for line in inFile:
#             if "THAT" in line:
#                if "fixed" in line:
#                     accept = True
#                     break
 #     print(accept)
      if True or accept:
          try:
            arguments = dict([x.split("=") for x in arguments[10:-1].split(", ")])
          except ValueError:
            print("VALUE ERROR", arguments)
            continue
          print(arguments)
          print(f)
          predictability_weight = arguments["predictability_weight"]
          deletion_rate = arguments["deletion_rate"]
          try:
           with open(PATH2+f, "r") as inFile:
             print("Opened", PATH2+f+"_Model")
             data = [x.split("\t") for x in inFile.read().strip().split("\n")]
             data = data[1:]
             for line in data:
                 if len(line) < len(header0):
                     if len(line) <= 4: # something is wrong
                       print("ERROR Something is wrong with this row", line)
                       continue
                     if len(line[4]) == 0:
                       print("ERROR Something is wrong with this row", line)
                       continue
                     assert len(line) == 5, line
                     try:
                       assert float(line[3]) < 50, line
                       assert float(line[4]) < 50, line
                     except ValueError:
                      print("ERROR", line)
                      continue
                     line.append("NA")
                 if len(line[0]) == 0:
                       print("ERROR Something is wrong with this row", line)
                       continue
                 if len(line[1]) == 0:
                       print("ERROR Something is wrong with this row", line)
                       continue
                 if len(line[2]) == 0:
                       print("ERROR Something is wrong with this row", line)
                       continue
                 if '"' in line[2]:
                       print("ERROR Something is wrong with this row", line)
                       continue

                 if "'" in line[2]:
                       print("Exclduing row with apostrophe to prevent downstream problems (affects two words in the dataset)", line)
                       continue
                 try:
                    assert int(line[5]) >= 0, line
                 except ValueError:
                     print("ERROR", line)
                     continue
                 if 0 in [len(x) for x in line]:
                      print("ERROR", line)
                      continue
         
                 try:
                   assert int(line[0]) < 200, line
                   if int(line[1]) > 24:
                     print("ERROR", line)
                   assert float(line[3]) < 200, line
                   assert float(line[4]) < 200, line
                 except ValueError:
                       print("ERROR Something is wrong with this row", line)
                       continue
                 assert suffix.startswith("script")
                 if len(line) > 6:
                       print("ERROR Something is wrong with this row", line)
                       continue
                 print("\t".join(line + [suffix, arguments["myID"], arguments["predictability_weight"], arguments["deletion_rate"], arguments["load_from_autoencoder"], arguments["load_from_plain_lm"]]), file=outFile)
          except FileNotFoundError:
             print("FAILED TO OPEN", f)
             pass
