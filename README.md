# ADEM (An Automatic Dialogue Evaluation Model)

## Basic information about ADEM
* Authors: Ryan Lowe, Michael Noseworthy, Iulian V. Serban, Nicolas Angelard-Gontier, 
        Yoshua Bengio, Joelle Pineau
* **ACL** 2017 Accepted Paper
* Under review as a conference paper at **ICLR** 2017
* Link: https://openreview.net/pdf?id=HJ5PIaseg

## Brief Introduction
ADEM is an automatic evaluation model for the quality of dialogue, aiming to capture the semantic similarity beyond word overlapping metrics (e.g BLEU, ROUGH, METOER) which correlating badly to human judgement, and calculate its score using extra information the context of conversation besides the reference response and model response. 

Learning the vector representations of dialogue context $\mathbf{c} \in \mathcal{R}^c$, model response $\hat{\mathbf{r}} \in \mathcal{R}^m$ and reference response $\mathbf{r} \in \mathcal{R}^r$ using a hierarchical RNN encoder, ADEM computes the score as follows:
    score() = ...
where M, N are learned parameters initialized with identity, $\alpha$, $\beta$ are scalar constants intialized in the range [0, 5]. The first and second term of the score function can be interpreted as the similarity of model response to context and reference response ,respectively in a linear transformation. 

ADEM is trained to minimize the model predictions an the human scores with L1 regularizations
    L ...
where \gamma is a scalar constant. The model is end to end differentiable and all parameters can be learned by backpropogation.
