# ExplainableAI
---

This project is to make some human-readible explainable metrics using XAI methods for pre-trained models for the study of light charged Higgs decaying to W and A bosons[ChargedHiggsAnalysis](https://github.com/choij1589/ChargedHiggsAnalysis) ($\mathrm{t} \rightarrow \mathrm{H^+b}, \mathrm{H^+} \rightarrow \mathrm{W^+A}, \mathrm{A} \rightarrow \mathrm{\mu^+\mu^-}$).

The procedure for running the scripts is as follows:
1. Load the input dataset. Each shuffling and train/valid/test split use the same random seeds as in the training stage, so the dataset should be no difference.
2. Load the pre-trained model in ```models``` directory.
3. Run the script in the ```scripts``` directory.
