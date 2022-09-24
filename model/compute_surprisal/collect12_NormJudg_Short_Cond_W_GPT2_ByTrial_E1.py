import glob
import codecs
import os

header = "Noun Item Region Condition Surprisal SurprisalReweighted ThatFraction ThatFractionReweighted".split(" ")
header += ["S1", "S2", "Word", "Script", "ID", "predictability_weight", "deletion_rate", "autoencoder", "lm"]

PATH0 = "/juice/scr/mhahn/reinforce-logs-both-short/results/"
PATH = "/juice/scr/mhahn/reinforce-logs-both-short/full-logs/"
PATH2 = "/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/"

with open(f"{PATH2}/{__file__}.tsv", "w") as outFile:
 print("\t".join(header), file=outFile)
 for f in sorted(os.listdir(PATH2)):
   shib = "12_NormJudg_Short_Cond_Shift_NoComma_Bugfix"
   if shib in f:
      suffix = "script_"+f[f.index(shib)+len(shib):f.index(".py")]
      if "_E1Stims_" not in f or "_OnlyLoc" in f or "ZERO" in f or "LE1" not in f:
        continue
      print(f)
      print(suffix)
      assert f.endswith("_Model"), f
      modelID = f.split("_")[-2]
      logPath = PATH0+"/*_"+modelID
      results_files = glob.glob(logPath)
      if len(results_files) == 0:
         print("ERROR26: NO RESULTS FILE", f)
         continue
      with codecs.open(results_files[0], "r", 'utf-8', "ignore") as inFile:
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
                 if len(line) == 10:
                    print("WARNING: COLUMN MISSING!!!", line)
                    if "L" in f:
                        print("ERROR this should not happen for this script", line, f)
                        continue
                    try:
                       _ = float(line[-1]) # Make sure the last entry is a number, as a basic sanity check
                    except:
                      print("ERROR 56", line)
                      continue
                    line.append("NA")
                 if len(line) != 11:
                    print("ERROR", line, 69)
                    continue
                 assert len(line) == 11, line
                 try:
                  if float(line[6]) > 100:
                    print("WARNING: INCORRECT PERCENTAGE in line[6]", line)
                 except ValueError:
                     print("ERROR", line, 76)
                     continue
                 assert float(line[7]) <= 100, line
                 try:
                    assert line[9] == "nan" or float(line[9]) <= 100, line
                 except ValueError:
                   print("ERROR", line, 82)
                   continue
                 try:
                   if float(line[4]) > 100:
                     print("ERROR", line, 85)
                     continue
                   if float(line[5]) > 100:
                     print("ERROR", line, 89)
                     continue
                 except ValueError:
                    print("ERROR", line, 92)
                    continue
#                    assert False, line
                 if line[3].startswith("V"):
                    print("ERROR something is wrong with this line", line)
                    continue
                 print("\t".join(line + [suffix, arguments["myID"], arguments["predictability_weight"], arguments["deletion_rate"], arguments["load_from_autoencoder"], arguments["load_from_plain_lm"]]), file=outFile)
          except FileNotFoundError:
             print("Couldn't open", PATH2+f+"_Model")
             pass
