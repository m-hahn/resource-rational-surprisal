from paths import WIKIPEDIA_HOME
import random
import gzip 

def load(language, partition="train", removeMarkup=True):
  if language == "english":
     path = WIKIPEDIA_HOME+"/english-"+partition+"-tagged.txt"
  elif language == "german":
    path = WIKIPEDIA_HOME+"/german-"+partition+"-tagged.txt.gz"
  chunk = []
  with open(path, "r") if language == "english" else gzip.open(path, "rb") as inFile:
    for line in inFile:
      if language == "german":
        line = line.decode()
      index = line.find("\t")
      if index == -1:
        if removeMarkup:
          continue
        else:
          index = len(line)-1
      word = line[:index]
      chunk.append(word.lower())
      if len(chunk) > 1000000:
      #   random.shuffle(chunk)
         yield chunk
         chunk = []
  yield chunk

def training(language):
  return load(language, "train")

def dev(language, removeMarkup=True):
  return load(language, "valid", removeMarkup=removeMarkup)

def test(language, removeMarkup=True):
  return load(language, "test", removeMarkup=removeMarkup)


