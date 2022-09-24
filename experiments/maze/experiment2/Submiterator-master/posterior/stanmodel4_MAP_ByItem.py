from collections import defaultdict
import math
import random
import torch

myID = random.randint(100,1000000)

LogRT = torch.FloatTensor([float(x.split("\t")[1]) for x in open("forStan/LogRT.tsv", "r").read().strip().split("\n")[1:]])
items = [int(x.split("\t")[1]) for x in open("forStan/items.tsv", "r").read().strip().split("\n")[1:]]
subjects = [int(x.split("\t")[1]) for x in open("forStan/subjects.tsv", "r").read().strip().split("\n")[1:]]
model_names = open("forStan/predictions.tsv", "r").read().strip().split("\n")[0].split("\t")
predictions = torch.FloatTensor([[float(q) for q in x.split("\t")[1:]] for x in open("forStan/predictions.tsv", "r").read().strip().split("\n")[1:]])
n_subjects = max(subjects)+1
n_items = max(items)+1
n_data = len(LogRT)
n_models = len(predictions[0])
dat = {}


print(LogRT.size(), predictions.size())
print( (LogRT.unsqueeze(1) * predictions).mean(dim=0) )
correlations = (((LogRT.unsqueeze(1) * predictions).mean(dim=0) - (LogRT.unsqueeze(1).mean(dim=0) * predictions.mean(dim=0)))/ torch.sqrt((LogRT.unsqueeze(1).pow(2).mean(dim=0, keepdim=True) - LogRT.unsqueeze(1).mean(dim=0, keepdim=True).pow(2)) * (predictions.unsqueeze(1).pow(2).mean(dim=0, keepdim=True) - predictions.unsqueeze(1).mean(dim=0, keepdim=True).pow(2)))).view(-1).tolist()
for i in range(n_models):
  print(model_names[i], "\t", correlations[i])
print(correlations)
#quit()

dat["LogRT"] = LogRT
dat["subjects"] = subjects
dat["predictions"] = predictions
dat["n_subjects"] = n_subjects
dat["n_data"] = n_data
dat["n_models"] = n_models


#print(dat)
parameters = {}
parameters["mix_logits"] = torch.zeros(n_models)
parameters["sigma"] = torch.FloatTensor([1.0])
parameters["slope"] = torch.FloatTensor([0.0])
parameters["intercept"] = torch.FloatTensor([0.0])


parameters["perParticipantEffects_slope"] = torch.zeros(n_subjects)
parameters["perParticipantEffects_intercept"] = torch.zeros(n_subjects)
parameters["sigma_perParticipantEffects_slope"] = torch.FloatTensor([1.0])
parameters["sigma_perParticipantEffects_intercept"] = torch.FloatTensor([1.0])

parameters["perItemEffects_slope"] = torch.zeros(n_subjects)
parameters["perItemEffects_intercept"] = torch.zeros(n_subjects)
parameters["sigma_perItemEffects_slope"] = torch.FloatTensor([1.0])
parameters["sigma_perItemEffects_intercept"] = torch.FloatTensor([1.0])


prior_ = torch.distributions.normal.Normal(0, 1, validate_args=None)

def model(parameters):
  log_density = 0
  log_density += prior_.log_prob(parameters["mix_logits"]).sum()
  log_density += prior_.log_prob(parameters["sigma"]).sum()
  log_density += prior_.log_prob(parameters["intercept"]).sum()
  log_density += prior_.log_prob(parameters["sigma_perItemEffects_intercept"]).sum()
  log_density += prior_.log_prob(parameters["perItemEffects_intercept"]/parameters["sigma_perItemEffects_intercept"].pow(2)).sum()
  log_density += prior_.log_prob(parameters["sigma_perItemEffects_slope"]).sum()
  log_density += prior_.log_prob(parameters["perItemEffects_slope"]/parameters["sigma_perItemEffects_slope"].pow(2)).sum()
  log_density += prior_.log_prob(parameters["sigma_perParticipantEffects_intercept"]).sum()
  log_density += prior_.log_prob(parameters["perParticipantEffects_intercept"]/parameters["sigma_perParticipantEffects_intercept"].pow(2)).sum()
  log_density += prior_.log_prob(parameters["sigma_perParticipantEffects_slope"]).sum()
  log_density += prior_.log_prob(parameters["perParticipantEffects_slope"]/parameters["sigma_perParticipantEffects_slope"].pow(2)).sum()
  mix = torch.nn.LogSoftmax()(parameters["mix_logits"])
#  print(predictions.size())
  random_intercept = parameters["perParticipantEffects_intercept"][subjects].unsqueeze(1) + parameters["perItemEffects_intercept"][items].unsqueeze(1)
  random_slope = parameters["perParticipantEffects_slope"][subjects].unsqueeze(1) + parameters["perItemEffects_slope"][items].unsqueeze(1)
  predicted = (6.5+parameters["intercept"] + random_intercept + (parameters["slope"] + random_slope) * predictions)
  likelihoods = torch.distributions.normal.Normal(0, parameters["sigma"]).log_prob(predicted - LogRT.unsqueeze(1))
  #print(likelihoods.size())
 # print(mix.unsqueeze(0))
#  print(likelihoods + mix.unsqueeze(0))
  likelihoods = torch.logsumexp(likelihoods + mix.unsqueeze(0), dim=1)
  likelihoods = likelihoods.sum()
#  print(likelihoods.size())
  return log_density+likelihoods, float(log_density), float(likelihoods)
#print(model(parameters))


for _, x in parameters.items():
    x.requires_grad=True

optim = torch.optim.Adam([x[1] for x in parameters.items()], lr=0.001)

learning_rate = 0.01

from collections import deque

likelihoods = deque(maxlen=1000)

for iteration in range(1000000):
   newLogLikelihood, log_prior, log_likelihood = model(parameters)
   print("Initializing at MAP parameters", iteration, newLogLikelihood, __file__, learning_rate, log_prior, log_likelihood, "ID", myID)
   optim.zero_grad()
   (-newLogLikelihood).backward()
   likelihoods.append(float(newLogLikelihood))
   if len(likelihoods) >= 999:
     oldLik = likelihoods.popleft()
#     print("Old and new likelihood", oldLik, float(newLogLikelihood))
     if float(newLogLikelihood) - oldLik < 1:
        learning_rate /= 2
        optim = torch.optim.Adam([x[1] for x in parameters.items()], lr=learning_rate)
        likelihoods = deque(maxlen=1000)

    # do optimization step
   optim.step()
   # printing
   if iteration % 100 == 0:
       mix_probs_ = list(zip(model_names, torch.nn.Softmax()(parameters["mix_logits"]).detach().numpy().tolist()))
       per_delta = defaultdict(float)
       per_lambda = defaultdict(float)
       for x, y in sorted(mix_probs_, key=lambda x:x[1]):
          _, delta, lambda_ = x.split("_")
          delta = float(delta)
          lambda_ = float(lambda_)
          print(delta, "\t", lambda_, "\t",y)
          per_delta[delta]+=y
          per_lambda[lambda_]+=y
       print(sorted(per_delta.items()))
       print(sorted(per_lambda.items()))
       
       print(parameters["sigma_perParticipantEffects_slope"])
       print(parameters["sigma_perParticipantEffects_intercept"])
       print(parameters["sigma"])
       print(parameters["intercept"])
       print(parameters["slope"])
   if (iteration+1) % 1000 == 0:
            torch.save({"params" : parameters, "models" : model_names}, f"/u/scr/mhahn/FITTING_RTs_FITS/{__file__}_{myID}.model")
            with open(f"/u/scr/mhahn/FITTING_RTs/{__file__}_{myID}.txt", "w") as outFile:
              print(iteration, newLogLikelihood, __file__, file=outFile)
              print("sigma", parameters["sigma"], file=outFile)
              print("intercept", parameters["intercept"], file=outFile)
              print("slope", parameters["slope"], file=outFile)
              print("prior", log_prior, "likelihood", log_likelihood, file=outFile)
   
