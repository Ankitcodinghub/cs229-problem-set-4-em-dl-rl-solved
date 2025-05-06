# cs229-problem-set-4-em-dl-rl-solved
**TO GET THIS SOLUTION VISIT:** [CS229 Problem Set #4-EM, DL, & RL Solved](https://www.ankitcodinghub.com/product/cs229-problem-set-4-em-dl-rl-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;96215&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS229 Problem Set #4-EM, DL, \u0026amp; RL Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
&nbsp;

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #4 2 1.&nbsp; Neural Networks: MNIST image classification

In this problem, you will implement a simple convolutional neural network to classify grayscale images of handwritten digits (0 – 9) from the MNIST dataset. The dataset contains 60,000 training images and 10,000 testing images of handwritten digits, 0 – 9. Each image is 28×28 pixels in size with only a single channel. It also includes labels for each example, a number indicating the actual digit (0 – 9) handwritten in that image.

The following shows some example images from the MNIST dataset: 1

The data for this problem can be found in the data folder as images train.csv, images test.csv, labels train.csv and labels test.csv.

The code for this assignment can be found within p1 nn.py within the src folder.

The starter code splits the set of 60,000 training images and labels into a sets of 59,600 examples

as the training set and 400 examples for dev set.

To start, you will implement a simple convolutional neural network and cross entropy loss, and train it with the provided data set.

The architecture is as follows:

<ol>
<li>(a) &nbsp;The first layer is a convolutional layer with 2 output channels with a convolution size of 4 by 4.</li>
<li>(b) &nbsp;The second layer is a max pooling layer of stride and width 5 by 5.</li>
<li>(c) &nbsp;The third layer is a ReLU activation layer.</li>
<li>(d) &nbsp;After the four layer, the data is flattened into a single dimension.</li>
<li>(e) &nbsp;The fith layer is a single linear layer with output size 10 (the number of classes).</li>
<li>(f) &nbsp;The sixth layer is a softmax layer that computes the probabilities for each class.</li>
<li>(g) &nbsp;Finally, we use a cross entropy loss as our loss function.</li>
</ol>
We have provided all of the forward functions for these different layers so there is an unambigious definition of them in the code. Your job in this assignment will be to implement functions that

1 https://commons.wikimedia.org/wiki/File:MnistExamples.png

</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #4 3 compute the gradients for these layers. However, here is some additional text that might be

helpful in understanding the forward functions.

We have discussed convolutional layers on the exam, but as a review, the following equation defines what we mean by a 2d convolution:

output[out channel, x, y] = convolution bias[out channel]+

</div>
</div>
<div class="layoutArea">
<div class="column">
􏰈

di,dj,in channel

</div>
<div class="column">
input[in channel, x + di, y + dy]∗

convolution weights[out channel, in channel, di, dj]

</div>
</div>
<div class="layoutArea">
<div class="column">
di and dj iterate through the convolution width and height respectively.

The output of a convolution is of size (# output channels, input width – convolution width + 1, output height – convolution height + 1). Note that the dimension of the output is smaller due to padding issues.

Max pooling layers simply take the maximum element over a grid. It’s defined by the following function

output[out channel, x, y] = max input[in channel, x ∗ pool width + di, y ∗ pool height + dy] di,dj

The ReLU (rectified linear unit) is our activation function. The ReLU is simply max(0, x) where x is the input.

We use cross entropy loss as our loss function. Recall that for a single example (x, y), the cross entropy loss is:

K

CE(y,yˆ)=−􏰈y logyˆ, kk

k=1

where yˆ ∈ RK is the vector of softmax outputs from the model for the training example x, and

y∈RK istheground-truthvectorforthetrainingexamplexsuchthaty=[0,…,0,1,0,…,0]⊤ contains a single 1 at the position of the correct class (also called a “one-hot” representation).

We are also doing mini-batch gradient descent with a batch size of 16. Normally we would iterater over the data multiple times with multiple epochs, but for this assignment we only do 400 batches to save time.

(a) [20 points]

Implement the following functions within p1 nn.py. We recommend that you start at the top of the list and work your way down:

i. backward softmax ii. backward relu

iii. backward log loss iv. backward linear

v. backward convolution vi. backward max pool

(b) [10 points] Now implement a function that computes the full backward pass. i. backward prop

</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #4 4 2. [15 points] Off Policy Evaluation And Causal Inference

In class we have discussed Markov decision processes (MDPs), methods for learning MDPs from data, and ways to compute optimal policies from that MDP. However, before we use that policy, we often want to get an estimate of the its performance. In some settings such as games or simulations, you are able to directly implement that policy and directly measure the performance, but in many situations such as health care implementing and evaluating a policy is very expensive and time consuming.

Thus we need methods for evaluating policies without actually implementing them. This task is usually referred to as off-policy evaluation or causal inference. In this problem we will explore different ways of estimating off policy performance and prove some of the properties of those estimators.

Most of the methods we discuss apply to general MDPs, but for the sake of this problem, we will consider MDPs with a single timestep. We consider a universe consisting of states S, actions A, a reward function R(s,a) where s is a state and a is an action. One important factor is that we often only have a subset of a in our dataset. For example, each state s could represent a patient, each action a could represent which drug we prescribe to that patient and R(s,a) be their lifespan after prescribing that drug.

A policy is defined by a function πi(s, a) = p(a|s, πi). In other words, πi(s, a) is the conditional probability of an action given a certain state and a policy.

We are given an observational dataset consisting of (s, a, R(s, a)) tuples.

Let p(s) denote the probability density function for the distribution of state s values within that dataset. Let π0(s,a) = p(a|s) within our observational data. π0 corresponds to the baseline policy present in our observational data. Going back to the patient example, p(s) would be the probability of seeing a particular patient s and π0(s,a) would be the probability of a patient receiving a drug in the observational data.

We are also given a target policy π1(s,a) which gives the conditional probability p(a|s) in our optimal policy that we are hoping to evaluate. One particular note is that even though this is a distribution, many of the policies that we hope to evaluate are deterministic such that given a particular state si, p(a|si) = 1 for a single action and p(a|s) = i for the other actions.

Our goal is to compute the expected value of R(s, a) in the same population as our observational data, but with a policy of π1 instead of π0. In other words, we are trying to compute:

E s∼p(s) R(s, a) a∼π1 (s,a)

Important Note About Notation And Simplifying Assumptions:

We haven’t really covered expected values over multiple variables such as E s∼p(s) R(s,a) in a∼π1 (s,a)

class yet. For the purpose of this question, you may make the simplifying assumption that our states and actions are discrete distributions. This expected value over multiple variables simply indicates that we are taking the expected value over the joint pair (s, a) where s comes from p(s) and a comes from π1(s,a). In other words, you have a p(s,a) term which is the probabilities of observing that pair and we can factorize that probability to p(s)p(a|s) = p(s)π1(s,a). In math notation, this can be written as:

</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #4

</div>
<div class="column">
5

</div>
</div>
<div class="layoutArea">
<div class="column">
E s∼p(s) a∼π1 (s,a)

</div>
<div class="column">
R(s, a) = 􏰈 R(s, a)p(s, a) (s,a)

= 􏰈 R(s, a)p(s)p(a|s) (s,a)

= 􏰈 R(s, a)p(s)π1(s, a) (s,a)

</div>
</div>
<div class="layoutArea">
<div class="column">
Unfortunately, we cannot estimate this directly as we only have samples created under policy π0 and not π1. For this problem, we will be looking at formulas that approximate this value using expectations under π0 that we can actually estimate.

We will make one additional assumption that each action has a non-zero probability in the observed policy π0(s, a). In other words, for all actions a and states s, π0(s, a) &gt; 0.

Regression: The simplest possible estimator is to directly use our learned MDP parameters to estimate our goal. This is usually called the regression estimator. While training our MDP, we learn an estimator Rˆ(s, a) that estimates R(s, a). We can now directly estimate

</div>
</div>
<div class="layoutArea">
<div class="column">
with

</div>
<div class="column">
E s∼p(s) R(s, a) a∼π1 (s,a)

E s∼p(s) Rˆ(s,a) a∼π1 (s,a)

</div>
</div>
<div class="layoutArea">
<div class="column">
If Rˆ(s, a) = R(s, a), then this estimator is trivially correct.

We will now consider alternative approaches and explore why you might use one estimator over

another.

(a) [2 points] Importance Sampling: One commonly used estimator is known as the impor- tance sampling estimator. Let πˆ0 be an estimate of the true π0. The importance sampling estimator uses that πˆ0 and has the form:

π1(s, a)

E s∼p(s) πˆ0(s,a)R(s,a)

a∼π0 (s,a)

Please show that if πˆ0 = π0, then the importance sampling estimator is equal to:

E s∼p(s) R(s, a) a∼π1 (s,a)

Note that this estimator only requires us to model π0 as we have the R(s, a) values for the items in the observational data. Answer:

(b) [2 points] Weighted Importance Sampling: One variant of the importance sampling es- timator is known as the weighted importance sampling estimator. The weighted importance sampling estimator has the form:

E π1 (s,a) R(s, a) s∼p(s) πˆ0 (s,a)

a∼π0 (s,a)

π1 (s,a)

E s∼p(s) πˆ0(s,a) a∼π0 (s,a)

</div>
</div>
</div>
<div class="page" title="Page 6">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #4 6 .

Please show that if πˆ0 = π0, then the importance sampling estimator is equal to: E s∼p(s) R(s, a)

a∼π1 (s,a)

Answer:

(c) [2 points] One issue with the weighted importance sampling estimator is that it can be biased in many finite sample situations. In finite samples, we replace the expected value with a sum over the seen values in our observational dataset. Please show that the weighted importance sampling estimator is biased in these situations.

Hint: Consider the case where there is only a single data element in your observational dataset. Answer:

<ol start="4">
<li>(d) &nbsp;[7 points] Doubly Robust: One final commonly used estimator is the doubly robust estimator. The doubly robust estimator has the form:
E s∼p(s) ((Ea∼π1(s,a)Rˆ(s,a))+ π1(s,a)(R(s,a)−Rˆ(s,a))) a∼π0(s,a) πˆ0(s, a)

One advantage of the doubly robust estimator is that it works if either πˆ0 = π0 or Rˆ(s, a) = R(s, a)

<ol>
<li>[4 points] Please show that the doubly robust estimator is equal to E s∼p(s) R(s,a) a∼π1 (s,a)
when πˆ0 = π0
</li>
<li>[3 points] Please show that the doubly robust estimator is equal to E s∼p(s) R(s,a)
a∼π1 (s,a)

when Rˆ(s, a) = R(s, a)
</li>
</ol>
Answer:
</li>
<li>(e) &nbsp;[2 points] We will now consider several situations where you might have a choice between the importance sampling estimator and the regression estimator. Please state whether the importance sampling estimator or the regression estimator would probably work best in each situation and explain why it would work better. In all of these situations, your states s consist of patients, your actions a represent the drugs to give to certain patients and your R(s, a) is the lifespan of the patient after receiving the drug.
<ol>
<li>[1 points] Drugs are randomly assigned to patients, but the interaction between the drug, patient and lifespan is very complicated.</li>
<li>[1 points] Drugs are assigned to patients in a very complicated manner, but the inter- action between the drug, patient and lifespan is very simple.</li>
</ol>
Answer:
</li>
</ol>
</div>
</div>
</div>
<div class="page" title="Page 7">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #4 7 3. [10 points] PCA

In class, we showed that PCA finds the “variance maximizing” directions onto which to project the data. In this problem, we find another interpretation of PCA.

Suppose we are given a set of points {x(1), . . . , x(m)}. Let us assume that we have as usual preprocessed the data to have zero-mean and unit variance in each coordinate. For a given unit-length vector u, let fu(x) be the projection of point x onto the direction given by u. I.e., if V = {αu : α ∈ R}, then

fu(x) = arg min ||x − v||2. v∈V

Show that the unit-length vector u that minimizes the mean squared error between projected points and original points corresponds to the first principal component for the data. I.e., show

</div>
</div>
<div class="layoutArea">
<div class="column">
that

</div>
<div class="column">
m

arg min 􏰈 ∥x(i) − fu(x(i))∥2 .

</div>
</div>
<div class="layoutArea">
<div class="column">
u:uT u=1 gives the first principal component.

Remark. If we are asked to find a k-dimensional subspace onto which to project the data so as to minimize the sum of squares distance between the original data and their projections, then we should choose the k-dimensional subspace spanned by the first k principal components of the data. This problem shows that this result holds for the case of k = 1.

Answer:

</div>
</div>
<div class="layoutArea">
<div class="column">
i=1

</div>
</div>
</div>
<div class="page" title="Page 8">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #4 8 4. [20 points] Independent components analysis

While studying Independent Component Analysis (ICA) in class, we made an informal argu- ment about why Gaussian distributed sources will not work. We also mentioned that any other distribution (except Gaussian) for the sources will work for ICA, and hence used the logistic distribution instead. In this problem, we will go deeper into understanding why Gaussian dis- tributed sources are a problem. We will also derive ICA with the Laplace distribution, and apply it to the cocktail party problem.

Reintroducing notation, let s ∈ Rn be source data that is generated from n independent sources. Let x ∈ Rn be observed data such that x = As, where A ∈ Rn is called the mixing matrix. We assume A is invertible, and W = A−1 is called the unmixing matrix. So, s = W x. The goal of ICA is to estimate W. Similar to the notes, we denote wjT to be the jth row of W. Note that this implies that the jth source can be reconstructed with wj and x, since sj = wjT x. We are given a training set {x(1) , . . . , x(m) } for the following sub-questions. Let us denote the entire training set by the design matrix X ∈ Rm×n where each example corresponds to a row in the matrix.

(a) [5 points] Gaussian source

For this sub-question, we assume sources are distributed according to a standard normal

distribution, i.e sj ∼ N (0, 1), j = {1, . . . , n}. The likelihood of our unmixing matrix, as described in the notes, is

</div>
</div>
<div class="layoutArea">
<div class="column">
mn j

</div>
</div>
<div class="layoutArea">
<div class="column">
l(W)=􏰈 log|W|+􏰈logg′(wTx(i)) , i=1 j=1

</div>
</div>
<div class="layoutArea">
<div class="column">
where g is the cumulative distribution function, and g′ is the probability density function of the source distribution (in this sub-question it is a standard normal distribution). Whereas in the notes we derive an update rule to train W iteratively, for the cause of Gaussian distributed sources, we can analytically reason about the resulting W.

Try to derive a closed form expression for W in terms of X when g is the standard normal CDF. Deduce the relation between W and X in the simplest terms, and highlight the ambiguity (in terms of rotational invariance) in computing W .

Answer:

<ol start="2">
<li>(b) &nbsp;[10 points] Laplace source.

For this sub-question, we assume sources are distributed according to a standard Laplace

distribution, i.e si ∼ L(0, 1). The Laplace distribution L(0, 1) has PDF fL (s) = 21 exp (−|s|). With this assumption, derive the update rule for a single example in the form

W := W + α (. . .) .

Answer:
</li>
<li>(c) &nbsp;[5 points] Cocktail Party Problem
For this question you will implement the Bell and Sejnowski ICA algorithm, but assuming a Laplace source (as derived in part-b), instead of the Logistic distribution covered in class. The file mix.dat contains the input data which consists of a matrix with 5 columns, with each column corresponding to one of the mixed signals xi. The code for this question can be found in p4 ica.py.

Implement the update W and unmix functions in p4 ica.py.
</li>
</ol>
</div>
</div>
</div>
<div class="page" title="Page 9">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #4 9

You can then run p4 ica.py in order to split the mixed audio into its components. The mixed audio tracks are written to midex i.wav in the output folder. The split audio tracks are written to split i.wav in the output folder.

To make sure your code is correct, you should listen to the resulting unmixed sources. (Some overlap or noise in the sources may be present, but the different sources should be pretty clearly separated.)

Note: In our implementation, we anneal the learning rate α (slowly decreased it over time) to speed up learning. In addition to using the variable learning rate to speed up convergence, one thing that we also do is choose a random permutation of the training data, and running stochastic gradient ascent visiting the training data in that order (each of the specified learning rates was then used for one full pass through the data).

</div>
</div>
</div>
<div class="page" title="Page 10">
<div class="layoutArea">
<div class="column">
CS229 Problem Set #4 10 5. [15 points] Markov decision processes

Consider an MDP with finite state and action spaces, and discount factor γ &lt; 1. Let B be the Bellman update operator with V a vector of values for each state. I.e., if V ′ = B(V ), then

V ′(s) = R(s) + γ max 􏰈 Psa(s′)V (s′).

a∈A

(a) [10 points] Prove that, for any two finite-valued vectors V1, V2, it holds true that

s∈S

</div>
</div>
<div class="layoutArea">
<div class="column">
s′ ∈S

</div>
</div>
<div class="layoutArea">
<div class="column">
||B(V1) − B(V2)||∞ ≤ γ||V1 − V2||∞. ||V ||∞ = max |V (s)|.

</div>
</div>
<div class="layoutArea">
<div class="column">
where

(This shows that the Bellman update operator is a “γ-contraction in the max-norm.”)

</div>
</div>
<div class="layoutArea">
<div class="column">
(b) [5 points] We say that V is a fixed point of B if B(V) = V. Using the fact that the Bellman update operator is a γ-contraction in the max-norm, prove that B has at most one fixed point—i.e., that there is at most one solution to the Bellman equations. You may assume that B has at least one fixed point.

Remark: The result you proved in part(a) implies that value iteration converges geometrically to the optimal value function V ∗. That is, after k iterations, the distance between V and V ∗ is at most γk.

Answer:

</div>
</div>
</div>
