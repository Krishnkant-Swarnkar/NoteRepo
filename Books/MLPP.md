# Machine Learning - A Probabilistic Perspective

-- Kevin P. Murphy



=================================================================================

# Chapter 1: Introduction

=================================================================================



**Machine Learning:**  A set of methods that can automatically detect patterns in data, and then use the uncovered patterns to predict future data, or to perform other kinds of decision making under uncertainty.

- ** interesting analogy between ML and STATs: [link](http://www-stat.stanford.edu/~tibs/stat315a/glossary.pdf)

- **Long tail property in data**: Few things are very common, but most things are quite rare.
  
- So, <u>generalization</u> from relatively small sample sizes is still important in the era of Big Data.
  
  

**ML (types): **

- Supervised/ Predictive Learning
  - Goal: Learn mapping from **x** (features/ attributes/ covariates  -- could be structured) to **y** (response variable)
    - Classification -- y is categorical
    - Regression -- y is real-valued
    - Ordinal -- y has natural ordering (eg: grades, product rating)
- Unsupervised/ Descriptive Learning
  - Goal: find interesting patterns in data
    - less well-defined. 
      - Not always sure what type of patterns to look for. 
      - No obvious error metric
- Reinforcement Learning
  - Goal: Learning to act when given occasional reward or punishment
  
  - Decision Theory -- serves as a basis of RL.
  
    

**Supervised Learning:**

- multi-label classification ~ multi-output classification

- Why probabilistic predictions? 
  - we learnt on limited training data
  - based on this limited knowledge we might not always be absolutely certain about the prediction.
  
- Often it is sufficient to know a single number $$p(y = 1|x, D)$$ in the binary case. Its implicitly $$p(y = 1|x, D,Model)$$.

- "I don't know" prediction is also important in some risk averse domains.

- Classification Eg: Spam filtering, classifying flowers, Handwriting Recognition, Image Recognition, Object detection and recognition, etc
  - Image classification: Some models are general purpose, but ignore some useful information (like natural ordering of pixels in images)
  - Invariance to some details is often required like (hairstyles in case of face detection/ spatial invariance) for solving specific tasks.
  
- Regression eg: stock market price prediction, age prediction of users, etc.

  

**Unsupervised Learning:**

- often called "Knowledge Discovery"

- Just given output data, no input data

- Unsup. learning ~ density estimation (unconditional), 

  - sup. learning ~ conditional density estimation

- Unsup. learning => multidimensional output variable => mutivariate probability density.

  - Sup. Learning => Single scalar output variable y => univariate conditional probability. (When not dealing with multiclass)
    
    Sup Learning generally simpler, coz we can use univariate probability models <u>(with input dependent parameters).</u>

- "Labeled data is not only expensive to acquire, but it also contains relatively little information, certainly not enough to reliably estimate the parameters of complex models."

- Two quotes by Geoffrey Hinton:
  > "*The number of samples require to train a large learning machine (for any task) depends on the amount of information that we ask it to predict*" -- Yann Le Cunn quoting Geoffery Hinton [(source)](https://youtu.be/Ount2Y4qxQo?t=1017)
  
  > *"When we’re learning to see, nobody’s telling us what the right answers are — we just look. Every so often, your mother says “that’s a dog”, but that’s very little information. You’d be lucky if you got a few bits of information — even one bit per second — that way. The brain’s visual system has 10^14 neural connections. And you only live for 10^9 seconds. So it’s no use learning one bit per second. You need more like 10^5  bits per second. And there’s only one place you can get that much information: from the input itself." — Geoffrey Hinton, 1996 (quoted in (Gorder 2006)).*
  
  - <mark>**Question 1.1:**</mark>
    
    *"Labeled data is not only expensive to acquire, but it also contains relatively little information, certainly not enough to reliably estimate the parameters of complex models."*. 
    
    The data in both supervised and unsupervised learning are (X,y) pairs. Then why is the information less in the case of supervised learning?
    
    **Ans:** Probably because we get way more unsupervised data per sec, as opposed to supervised.
  
- <u>Discovering Clusters</u>
  
  - Data ---> possibly many clusters (not sure how many, let it be K).
  
  - Unsupervised Case --> we're free to choose K
  
    Supervised Case --> it was 2 classes (for binary classification)
  
  - Goal:
  
    1. to find K. :: $$K^* = \arg \max_K \,\, p(K | \mathcal D)$$
  
    2. to estimate which cluster each data point belongs to. $$z_i^* = \arg\max_k \,\,p(z_i=k|x_i,\mathcal D),\,\,\, z_i \in \{1,2,...K\}$$
  
  - *Model Based Clustering*: As opposed to Ad Hoc clustering, this approach fits a probabilistic model to the data.
  
  - Applications: astronomy (cluster celestial bodies), e-commerce (cluster users, target ads),  biology (cluster cell sub populations)
  
- <u>Discovering Latent Factors</u>

  - *Dimensionality reduction* is useful with higher dimensional data
    - reduce the dimensionality by projecting the data to a lower dimensional subspace which captures the “essence” of the data.
  - Motivation: Apparently High Dimensional data may have small number of degrees of variability (corresponding to *latent factors*)
  - Low dimension: 
    - statistical models with better predictive accuracy
    - fast nearest neighbor searches
    - better visualization
  - Example: PCA (Principle Component Analysis)
    - Kinda Unsupervised multi-output linear regression (invert the arrow in $$z \rightarrow y, \, \, y=\text{data}, z=\text{latent cause}$$). 
    - Applications: Biology (gene micro-array data interpretation), NLP (Latent Semantic Analysis -- actually SVD), Signal Processing (ICA -- Independent Component Analysis -- a PCA variant -- for blind Source Separation), Computer Graphics (creating animations)

- <u>Discovering Graph Structure</u>

  - Learning "dependence between variables" from the data. Create a graph G out of it.
    - $$\hat G = \arg\max p(G|\mathcal D)$$
  - Two applications: 
    1. Discover Knowledge (-- the graph) from data (-- unstructured data)
    2. Get better joint probability estimators
       - to better model correlations/ make predictions

- Data Imputation => (<u>Matrix Completion</u>)

  - Inferring plausible values for the missing entries in the data (aka. *Imputation*).
  
  - Examples: Image inpainting, Collaborative filtering (Netflix Prize competition), Market basket analysis (generally a binary matrix)
  
  - Possible approach -- fitting a joint probability model (though less interpretable).
  
    

**Basic ML Concepts :**

- <u>Parametric</u> Models: probabilistic models with fixed no. of parameters.

  ​								faster, but make stronger assumptions about nature of data distribution.

  <u>Non-Parametric Models</u>: probabilistic models where #parameters grow with training data.

  ​								more flexible but computationally intractable for large datasets.

- <u>KNN Classifier</u> (K nearest Neighbor) -- Non-Parametric Model

  - $$p(y=c|x,\mathcal D, K) = \frac{1}{K} \sum_{i\in N_K(x,\mathcal D)} \mathbb I(y_i = c)$$
  - where $$N_K(x,\mathcal D)$$ denotes the indices of K nearest points to x in $$\mathcal D$$
  - Voronoi Tessellation: partition of space based on 1NN which associates region $$V(x_i)$$ -- all points closest to $$x_i$$ than any other point -- for each point $$x_i$$.
  - **Note:** KNN does not work well with high dimensional inputs. Reason follows...

- <u>Curse of Dimensionality</u>

  - The local neighborhood of the points ('f' **percent** data in the proximity of the given point) becomes more and more spacious, as we go higher in the number of dimensions.  This effect is more pronounced if f is smaller

    <img src="https://i.stack.imgur.com/L3fF3.png=0.1x" alt="MLPP illustration of curse of dimensionality" style="zoom:40%;" /> 

    Intuitively, take a line ($$1 m$$), a square ($$1 m^2$$), and a cube ($$1 m^3$$).

    ​	1% of points on line => 0.01 $$m$$

    ​	1% of points on square => 0.1 $$m$$ on each axes

    ​	1% of points on cube => 0.215 $$m$$ on each axes

  - **Issue:** In higher dimensions, even the nearest neighbors are not so near,

    - And hence they might not be good predictors about the behavior of input-output function at a point.
      - Which probably indicated the need of a lot more data, when we go in higher dimensions.

- <u>Parametric Models for Classification and Regression.</u>

  - To fight this curse of dimensionality, often, parametric models are used
  
    - they make some assumptions about the nature of data distribution ( $$\text{either}\, p(y|x)\, \text{or}\, p(x) $$ ).
      - called inductive biases.
  
  - Eg-1: Linear regression
  
    - response is linear function of inputs $$y(x)=w^Tx+\epsilon,\,\,\,where\,,\epsilon \sim \mathcal N(\mu, \sigma^2)$$
  
      which renders the conditional prob (under some simple assumptions) as : 
      
      ​	$$p(y|x,\theta) = \mathcal N(y|w^T\phi(x),\sigma^2)$$
  
  - Eg-2: Logistic Regression
  
    - Since the response is binary, we make the Bernoulli distribution assumption:
  
      - $$p(y|x,w) = Ber(y|\mu(x)) , \,\,\,\mu(x) = p(y=1|x) = sigm(w^T\phi(x))$$
  
      - Decision rule: $$p(y=1|x)>0.5 \implies \hat y(x)=1 $$
  
        If data is not linearly separable, this decision rule gives a non-zero training error.
  
  - The Gaussian / Bernoulli distribution assumption is the inductive bias (consider contrasting it to KNN regression)
  
- <u>Overfitting</u>

  - We should avoid trying to model every minor variation in the input, since this is more likely to be noise than true signal.

- <u>Model Selection</u>

  - Find the model with least generalization error (expected error on future unseen data).
  - Use validation set. Use it to select the best model, refit it to the training data.
    - When small data --- less number of examples (we might not've enough data to train/ validate):
      - use K fold cross validation.
    - 

- <u>No Free Lunch Theorem</u>

  _All models are wrong, but some models are useful._ — George Box 

  - There is no universally best model.
    - we make assumptions, which work well on one domain, poorly on other.
  - From another perspective:
    - any two algorithms are equivalent when their performances are averaged across all possible problems.
  
  
  
  





=================================================================================

# Chapter 2: Probability

================================================================================= 



