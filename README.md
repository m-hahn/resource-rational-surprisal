# A Resource-Rational Model of Human Processing of Recursive Linguistic Structure

This repository contains all code and data for the paper.

- [Experiments: Code, Data, and Analyses](experiments/)
- [Model: Code and Results](model/)
- [Materials: Experimental Items and Nouns](materials/)

Models and log files (linked using local paths in the scripts) are publicly available [here](https://nlp.stanford.edu/~mhahn/resource-rational-surprisal/). Model output on the critical stimuli, for reproducing simulation results, has been published [at Zenodo](https://zenodo.org/record/6988696).
Models have also been published [at Zenodo](https://zenodo.org/record/6602698).
The corpus used for model fitting is available [here](https://nlp.stanford.edu/~mhahn/tabula-rasa/DATA/), specifically [training](https://nlp.stanford.edu/~mhahn/tabula-rasa/DATA/english-train-tagged.txt) and [held-out](https://nlp.stanford.edu/~mhahn/tabula-rasa/DATA/english-valid-tagged.txt) partitions. The corpus was [previously](https://doi.org/10.1162/tacl_a_00306) derived from the official English Wikipedia dump by applying [Wikiextractor](http://attardi.github.io/wikiextractor/) and [TreeTagger](https://cis.uni-muenchen.de/~schmid/tools/TreeTagger/). An example is given [here](materials/corpus_example.txt). Refer [here](model/compute_surprisal/README.md) if you want to run the model on your own stimuli.

## Links to Figures in Paper and SI

Main Paper, Figure 3:


- [Experiment 1, Surprisal Theory](model/compute_surprisal/analyze_output/figures/model-critical-experiment1-005-points_SQUARE_Bits.pdf)
- [Experiment 1, Model](model/compute_surprisal/analyze_output/figures/model-critical-experiment1-05-points_Bits.pdf)
- [Experiment 1, Reading Times](experiments/maze/experiment1/Submiterator-master/figures/logRT-points-fit_errorbars_noLogTransform.pdf)
- [Experiment 1, Analysis Script](experiments/maze/experiment1/Submiterator-master/analyze.R)
- [Experiment 2, Surprisal Theory](model/compute_surprisal/analyze_output/figures/model-critical-experiment2-005-points_SQUARE.pdf)
- [Experiment 2, Model](model/compute_surprisal/analyze_output/figures/model-critical-experiment2-05-points_Bits.pdf)
- [Experiment 2, Reading Times](experiments/maze/experiment2/Submiterator-master/figures/logRT-points-fit_errorbars_noLogTransform.pdf)
- [Experiment 2, Analysis Script](experiments/maze/experiment2/Submiterator-master/analyze_Experiment2.R)

Main Paper, Figure 4:


- [English](experiments/production/experiment3_english/Submiterator-master/figures/rates_by_conditional.pdf)
- [Spanish](experiments/production/experiment3_spanish/Submiterator-master/figures/rates_by_conditional.pdf)
- [German](experiments/production/experiment3_german/Submiterator-master/figures/rates_by_conditional.pdf)




Model Predictions (SI Appendix, Section 2):

- [Retention Probabilities](model/compute_surprisal/figures/retention_rates_lambda1_20_raw_overall.pdf)
- [Retention Probabilities (that and prepositions)](model/compute_surprisal/figures/retention_rates_lambda1_20_raw_overall_functionWords.pdf)
- [Model Surprisal (Experiment 1)](model/compute_surprisal/analyze_output/figures/model-critical-experiment1-full-NoLimit_Lambda1_Integer_Bits.pdf)
- [Model Surprisal (Experiment 2)](model/compute_surprisal/analyze_output/figures/model-critical-experiment2-full-NoLimit_Lambda1_Integer_Bits.pdf)
- [Model Fit (AIC)](experiments/maze/meta/figures/analyze_Model_IncludingNoTP_E12_Viz_R_AICRaw_Lambda1_Integer.pdf)
- [Scaling of Surprisal and RTs](experiments/maze/meta/figures/analyze_Model_PlotForExpt12_Joint_ModelHuman_OnlyExpt12_R_Bits.pdf)


Analyses for RT Studies (SI Appendix, Section 3):

- [Experiment 1, Main effects](experiments/maze/experiment1/Submiterator-master/figures/posterior-histograms-main_effects.pdf) 
- [Experiment 1, Interactions](experiments/maze/experiment1/Submiterator-master/figures/posterior-histograms-interactions.pdf)
- [Experiment 1, Effects in raw RTs](experiments/maze/experiment1/Submiterator-master/figures/posterior-histograms-RawEffects.pdf)
- [Experiment 2, Main effects](experiments/maze/experiment2/Submiterator-master/figures/posterior-histograms-main_effects.pdf) 
- [Experiment 2, Interactions](experiments/maze/experiment2/Submiterator-master/figures/posterior-histograms-interactions.pdf)
- [Experiment 2, Effects in raw RTs](experiments/maze/experiment2/Submiterator-master/figures/posterior-histograms-RawEffects.pdf)
- [Errors by position](experiments/maze/meta/figures/errors-by-position.pdf)
- [Error rates by participant](experiments/maze/meta/figures/slides-errors.pdf)
- [Analysis of errors: Main effects](experiments/maze/meta/figures/analyze_Errors_R_posterior-histograms-main_effects.pdf) 
- [Analysis of errors: Interactions](experiments/maze/meta/figures/analyze_Errors_R_posterior-histograms-interactions.pdf)
- [Analysis including incorrect responses: Main effects](experiments/maze/meta/figures/analyze_WithErrors_R_posterior-histograms-main_effects.pdf) 
- [Analysis including incorrect responses: Interactions](experiments/maze/meta/figures/analyze_WithErrors_R_posterior-histograms-interactions.pdf)
- [Analysis including incorrect responses: Effects involving answer correctness](experiments/maze/meta/figures/analyze_WithErrors_R_posterior-histograms-interactionsWithCorrect.pdf)

Analyses for Production Studies (SI Appendix, Section 4):

- [English](experiments/production/experiment3_english/Submiterator-master/figures/posterior-histograms.pdf) 
- [German](experiments/production/experiment3_german/Submiterator-master/figures/posterior-histograms.pdf) 
- [Spanish](experiments/production/experiment3_spanish/Submiterator-master/figures/posterior-histograms.pdf) 


Ratings Studies (SI Appendix, Section 5):

- [Ratings Study 1, raw](experiments/rating/study1/Submiterator-master/figures/rating_understand-logodds-byNoun-LogRatio.pdf)
- [Ratings Study 1, analysis](experiments/rating/study1/Submiterator-master/figures/posterior-histograms-main_effects.pdf)
- [Ratings Study 2, raw](experiments/rating/study2/Submiterator-master/figures/rating_understand-logodds-byNoun-LogRatio.pdf)
- [Ratings Study 2, analysis](experiments/rating/study2/Submiterator-master/figures/posterior-histograms-main_effects.pdf)

Previous Reading Time Studies (SI Appendix, Section 6):

- [Experiment S1, raw](experiments/maze/previous/study1_EmbeddingBias/Submiterator-master/figures/rt-raw.pdf)
- [Experiment S1, analysis](experiments/maze/previous/study1_EmbeddingBias/Submiterator-master/figures/posterior-histograms-main_effects.pdf)
- [Experiment S2, raw](experiments/maze/previous/study2_compatible/Submiterator-master/figures/rt-raw.pdf)
- [Experiment S2, analysis](experiments/maze/previous/study2_compatible/Submiterator-master/figures/posterior-histograms-main_effects.pdf)
- [Experiment S3, raw](experiments/maze/previous/study3_OneTwo/Submiterator-master/figures/logRT-points-fit_NoLogTransform.pdf) 
- [Experiment S3, analysis](experiments/maze/previous/study3_OneTwo/Submiterator-master/figures/posterior-histograms.pdf)
- [Experiment S4, raw](experiments/maze/previous/study4_compatibility/Submiterator-master/figures/logRT-points-fit_NoLogTransform.pdf) 
- [Experiment S4, analysis](experiments/maze/previous/study4_compatibility/Submiterator-master/figures/posterior-histograms.pdf)
- [Experiment S5, raw](experiments/maze/previous/study5_replication/Submiterator-master/figures/logRT-points-fit_errorbars_noLogTransform.pdf) 
- [Experiment S5, analysis: main effects](experiments/maze/previous/study5_replication/Submiterator-master/figures/posterior-histograms-main_effects.pdf) 
- [Experiment S5, analysis: interactions](experiments/maze/previous/study5_replication/Submiterator-master/figures/posterior-histograms-interactions.pdf)

Meta-Analysis (SI Appendix, Section 6.6):

- [Main Effects](experiments/maze/meta/figures/posterior-histograms-main_effects.pdf)
- [Interactions](experiments/maze/meta/figures/posterior-histograms-interactions.pdf)
- [In raw reading times](experiments/maze/meta/figures/posterior-histograms-RawEffects.pdf)
- [Main Effects (with regularizing prior)](experiments/maze/meta/figures/posterior-histograms-main_effects_prior.pdf)
- [Interactions (with regularizing prior)](experiments/maze/meta/figures/posterior-histograms-interactions_prior.pdf)
- [In raw reading times (with regularizing prior)](experiments/maze/meta/figures/posterior-histograms-RawEffects_prior.pdf)

Model Fit in Fillers (SI Appendix, Section 9):

- [Maze](model/compute_surprisal/analyze_output/fillers/figures/analyzeFillers_freq_BNC_Spillover_Averaged_New_R_Lambda1_Integer.pdf)
- [Eye Tracking: First Pass Times](model/compute_surprisal/analyze_output/fillers/figures/analyzeFillers_freq_BNC_FPRT_Spillover_Averaged_New_R_Lambda1_Integer.pdf)
- [Self-Paced Reading](model/compute_surprisal/analyze_output/fillers/figures/analyzeFillers_freq_BNC_SPR_Spillover_Averaged_New_R_Lambda1_Integer.pdf)

Control Studies for Embedding Bias (SI Appendix, Section 10):

- [Posterior with Wikipedia Counts](experiments/maze/meta/controls/figures/posterior-histograms-EmbeddingBias_wikipedia.pdf)
- [Posterior with ukWaC Counts](experiments/maze/meta/controls/figures/posterior-histograms-EmbeddingBias_ukwac.pdf)
- [Posterior with COCA Counts](experiments/maze/meta/controls/figures/posterior-histograms-EmbeddingBias_coca.pdf)
- [Raw data, Experiment 3 (Wikipedia)](experiments/production/experiment3_english/Submiterator-master/figures/rates_by_conditional.pdf)
- [Raw data, Experiment 3 (ukWaC)](experiments/production/experiment3_english/Submiterator-master/figures/rates_by_conditional_ukwac.pdf)
- [Raw data, Experiment 3 (COCA)](experiments/production/experiment3_english/Submiterator-master/figures/rates_by_conditional_COCA.pdf)
- [Fixed effect posterior, Experiment 3 (Wikipedia)](experiments/production/experiment3_english/Submiterator-master/figures/posterior-histograms.pdf)
- [Fixed effect posterior, Experiment 3 (ukWaC)](experiments/production/experiment3_english/Submiterator-master/figures/posterior-histograms_ukwac.pdf)
- [Fixed effect posterior, Experiment 3 (COCA)](experiments/production/experiment3_english/Submiterator-master/figures/posterior-histograms_COCA.pdf)
- [Model fit by predictor](experiments/maze/meta/controls/analyze_Previous_AIC_Single_R.pdf)
- [Embedding bias and RT effect](experiments/maze/meta/output/plotNounIntercepts_R.pdf)

Nouns (SI Appendix, Section 11):

- [German](materials/nouns/figures/nouns_german.pdf)
- [Spanish](materials/nouns/figures/nouns_spanish.pdf)
- [English (Wikipedia)](materials/nouns/English/figures/All_nouns_byType.pdf)
- [English (ukWaC)](materials/nouns/English/figures/All_nouns_byType_ukWaC.pdf)
- [English (COCA)](materials/nouns/English/figures/All_nouns_byType_COCA.pdf)


