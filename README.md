# Towards An Automatic Turing Test: Learning to Evaluate Dialogue Responses 
### A Tensorflow Implementation of ADEM - An Automatic Dialogue Evaluation Model
## Basic information about ADEM
* Authors: Ryan Lowe, Michael Noseworthy, Iulian V. Serban, Nicolas Angelard-Gontier, 
        Yoshua Bengio, Joelle Pineau
* [**ACL** 2017 Accepted Paper](https://chairs-blog.acl2017.org/2017/04/05/accepted-papers-and-demonstrations/)
* Under review as a conference paper at **ICLR** 2017
* Link: https://openreview.net/pdf?id=HJ5PIaseg

## Brief Introduction
ADEM is an automatic evaluation model for the quality of dialogue, aiming to capture the semantic similarity beyond word overlapping metrics (e.g BLEU, ROUGH, METOER) which correlating badly to human judgement, and calculate its score using extra information the context of conversation besides the reference response and model response. 

Learning the vector representations of dialogue context $\mathbf{c} \in \mathcal{R}^c$, model response $\hat{\mathbf{r}} \in \mathcal{R}^m$ and reference response $\mathbf{r} \in \mathcal{R}^r$ using a hierarchical RNN encoder, ADEM computes the score as follows:

$$\text{score}(c, r, \hat{r}) = (\mathbf{c}^TM\hat{\mathbf{r}}+\mathbf{r}^TN\hat{\mathbf{r}} -\alpha) / \beta$$

where M, N are learned parameters initialized with identity, $\alpha$, $\beta$ are scalar constants intialized in the range [0, 5]. The first and second term of the score function can be interpreted as the similarity of model response to context and reference response ,respectively in a linear transformation. 

ADEM is trained to minimize the model predictions an the human scores with L1 regularizations

$$\mathcal{L} = \sum_{i=1:K}[{\text{score}(c_i, r_i, \hat{r_i}) - human\_score_i}]^2 + \gamma \|\theta\|_1$
    where $\theta = \{M, N\}$$

where \gamma is a scalar constant. The model is end to end differentiable and all parameters can be learned by backpropogation.


## Usage
#### 1. ADEM
If you have the vector representation of dialogue, model response and reference response already, just feed them into model ADEM.
Different dimension of vectors are supported.
```python
   from ADEM import ADEM
   model = ADEM(context_dim, model_response_dim, reference_response_dim, learning_rate)
   model.train_on_single_batch(train_session, context, model_response, reference_response, human_score)
```
#### 2. ADEM with encoder
If you DO NOT have the vector representation of dialogue, model response, reference response, you can input your data which have been mapped from text to word index (or character index) into ADEMWithEncoder. mask....(Since length must be the same in...)

 
