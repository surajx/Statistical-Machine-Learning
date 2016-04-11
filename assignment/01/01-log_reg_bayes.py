
# coding: utf-8

# In[1]:

# Classification with Logistic Regression, Bayes Linear Regression


# ###### COMP4670/8600 - Introduction to Statistical Machine Learning - Assignment 1 (due: Monday, 18 April, 23:59)

# Name: Suraj Narayanan Sasikumar
#
# Student ID: u5881495

# ## Instructions
#
# |             |Notes|
# |:------------|:--|
# |Maximum marks| 20|
# |Weight|20% of final grade|
# |Format| Complete this ipython notebook. Do not forget to fill in your name and student ID above|
# |Submission mode| Use [wattle](https://wattle.anu.edu.au/)|
# |Formulas| All formulas which you derive need to be explained unless you use very common mathematical facts. Picture yourself as explaining your arguments to somebody who is just learning about your assignment. With other words, do not assume that the person marking your assignment knows all the background and therefore you can just write down the formulas without any explanation. It is your task to convince the reader that you know what you are doing when you derive an argument. Typeset all formulas in $\LaTeX$.|
# | Code quality | Python code should be well structured, use meaningful identifiers for variables and subroutines, and provide sufficient comments. Please refer to the examples given in the tutorials. |
# | Code efficiency | An efficient implementation of an algorithm uses fast subroutines provided by the language or additional libraries. For the purpose of implementing Machine Learning algorithms in this course, that means using the appropriate data structures provided by Python and in numpy/scipy (e.g. Linear Algebra and random generators). |
# | Late penalty | For every day (starts at midnight) after the deadline of an assignment, the mark will be reduced by 5%. No assignments shall be accepted if it is later than 10 days. |
# | Coorperation | All assignments must be done individually. Cheating and plagiarism will be dealt with in accordance with University procedures (please see the ANU policies on [Academic Honesty and Plagiarism](http://academichonesty.anu.edu.au)). Hence, for example, code for programming assignments must not be developed in groups, nor should code be shared. You are encouraged to broadly discuss ideas, approaches and techniques with a few other students, but not at a level of detail where specific solutions or implementation issues are described by anyone. If you choose to consult with other students, you will include the names of your discussion partners for each solution. If you have any questions on this, please ask the lecturer before you act. |
# | Solution | To be presented in the tutorials. |

# This assignment has two parts. In the first part, you apply logistic regression to given data (maximal 13 marks). In the second part, you answer a number of questions (maximal 7 marks). All formulas and calculations which are not part of Python code should be written using $\LaTeX$.

# $\newcommand{\dotprod}[2]{\left\langle #1, #2 \right\rangle}$
# $\newcommand{\onevec}{\mathbb{1}}$
#
# Setting up the environment

# In[2]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import scipy.optimize as opt
import scipy.stats as stats

#get_ipython().magic('matplotlib inline')


# ## The data set
#
#
# The data set contains mass-spectrometric data which are used to distinguish between cancer and normal patterns (https://archive.ics.uci.edu/ml/datasets/Arcene).
#
# Please download the following data:
# * training data https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_train.data,
# * training labels https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_train.labels,
# * validation data https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_valid.data, and
# * validation labels https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/arcene_valid.labels.
#
# The following code reads the training and validation data into your workspace.

# In[3]:

X_train = np.loadtxt("arcene_train.data")
y_train = np.loadtxt("arcene_train.labels")
X_val   = np.loadtxt("arcene_valid.data")
y_val   = np.loadtxt("arcene_valid.labels")

print(X_train.shape)
print(X_val.shape)


# As the 10000-dimensional input space might lead to long computation times, we have prepared a subset of 220 features. Unless otherwise stated, you will from now on work only with these subset of the training and validation data.

# In[4]:

feature_mask = np.array(
    [   4,   37,   85,  187,  233,  255,  375,  413,  435,  468,
      470,  477,  519,  528,  628,  629,  643,  661,  678,  682,
      750,  783,  786,  802,  936,  997, 1035, 1043, 1113, 1186,
     1288, 1294, 1316, 1319, 1441, 1475, 1488, 1546, 1577, 1589,
     1666, 1671, 1739, 1761, 1830, 1848, 1881, 1882, 1975, 2057,
     2116, 2157, 2170, 2297, 2308, 2406, 2407, 2460, 2532, 2619,
     2632, 2634, 2644, 2656, 2697, 2717, 2771, 2817, 2865, 2937,
     3006, 3033, 3109, 3250, 3256, 3364, 3369, 3386, 3517, 3574,
     3611, 3643, 3660, 3702, 3777, 3783, 3856, 4008, 4016, 4036,
     4058, 4081, 4147, 4157, 4182, 4197, 4202, 4230, 4251, 4305,
     4379, 4440, 4454, 4467, 4485, 4555, 4557, 4579, 4585, 4607,
     4685, 4702, 4721, 4730, 4894, 4899, 4954, 4959, 5004, 5048,
     5076, 5200, 5230, 5242, 5249, 5306, 5355, 5472, 5476, 5631,
     5720, 5773, 5790, 5936, 5994, 6106, 6111, 6162, 6163, 6192,
     6304, 6350, 6402, 6407, 6439, 6462, 6480, 6494, 6522, 6555,
     6596, 6620, 6678, 6773, 6791, 6869, 6888, 6889, 6904, 6927,
     6957, 6961, 7101, 7196, 7214, 7271, 7279, 7297, 7425, 7431,
     7436, 7462, 7505, 7512, 7627, 7651, 7747, 7793, 7812, 7855,
     7856, 7860, 7866, 7932, 7976, 7993, 8006, 8131, 8155, 8257,
     8266, 8270, 8367, 8378, 8440, 8472, 8501, 8726, 8761, 8829,
     8831, 8903, 9021, 9024, 9026, 9060, 9081, 9116, 9211, 9214,
     9233, 9319, 9371, 9506, 9539, 9549, 9603, 9616, 9633, 9703])
X_train_sub = X_train[:, feature_mask]
X_val_sub   = X_val[:, feature_mask]

y_train[y_train==-1] = 0
y_val[y_val==-1] = 0

print(X_train_sub.shape)
print(X_val_sub.shape)


# ## (2 points) Normalise the input data
# Find a linear transformation of the training data resulting in a zero mean and unit variance. Report the parameters of the linear transformation for the first ten dimensions of the input data.
#
# In general:
# * Under which circumstances does working with this transformed data lead to an advantage?
# * When is it counterproductive to normalise input data to zero mean and/or to unit variance?

# In[5]:

def normalize(X, mu, sd):
    return (X - mu)/sd

X_train_sub_mean = [ X_train_sub[:,col].mean() for col in np.arange(X_train_sub.shape[1])]
X_train_sub_sd = [ X_train_sub[:,col].std() for col in np.arange(X_train_sub.shape[1])]
print("Mean for first 10 dimension:", X_train_sub_mean[:10])
print("Standard Deviation for first 10 dimension:", X_train_sub_sd[:10])
X_train_sub_norm = normalize(X_train_sub, X_train_sub_mean, X_train_sub_sd)
X_val_sub_norm = normalize(X_val_sub, X_train_sub_mean, X_train_sub_sd)


# ### Solution
#
# ##### Circumstances where normalization is an Advantage
# * When features are represented in different scale. Since the error function uses an euclidean distance to calculate erro in prediction, the features with larger value (due to it's scale) get more influence on prediction when compared to features whose numerical values are small.
# * When there are outliers in the data
# * When we know that the data has a normal distribution
# * When we know we are going to use optimation algorithms like gradient discent which converge much faster when the data is normalized.
#
# ##### Circumstances where normalization is Counter-productive
# * When the input distribution is known to be not gaussian
# * When all the features are in the same range, there by rendering normalization useless
# * When some features are known to influence more on the prediction of the target normalization is counter-productive.

# ## (1 point) Error function for logistic regression with quadratic regularisation
# Define a Python function calculating the *cross-entropy* $ E(\mathbf{w}) = - \ln p(\mathbf{t} \;|\; \mathbf{w}) $for logistic regression with two classes and quadratic (l2) regularisation for a given parameter vector $\mathbf{w}$.

# In[6]:

def sig(X):
    """S shaped function, known as the sigmoid"""
    return 1 / (1 + np.exp(- X))

def coss_entropy_error_fn(w, X, t, l):
    p_1 = sig(np.dot(w, X.T))
    erfn_1 = np.dot(t, np.log(p_1))
    erfn_2 = np.dot((1-t), np.log(1 - p_1))
    cost = -1 * (erfn_1 + erfn_2) + (l/2) * np.dot(w.T, w)
    return cost.mean()


# ## (1 point) Gradient for logistic regression with quadratic regularisation
# Define a Python function calculating the gradient of the above *cross-entropy* for logistic regression with two classes and quadratic (l2) regularisation.

# In[7]:

def grad(w, X, t, l):
    p_1 = sig(np.dot(w, X.T))
    err = p_1 - t
    grd = np.dot(err, X) + l * w
    return grd



# ## (1 point) Finding the optimal parameters for given regularisation
# Using the error function and the gradient defined above, you now setup the optimisation finding the optimal parameter vector $\mathbf{w}^\star$ for the training data and a fixed regularisation constant $\lambda $.
# Use the function *scipy.optimize.fmin_bfgs* as optimiser.
#
# For each $\lambda= 10^k, k=-3,-2,..,1$, report the first 10 components of the optimal parameter $\mathbf{w}^\star$ found.

# In[8]:

def add_ones(X):
    N = X.shape[0]
    D = X.shape[1]
    M = D + 1
    tmp = np.ones((N,M))
    tmp[:,:-1] = X
    return tmp

def train(X, t, l, disp=1):
    X = add_ones(X)
    M = X.shape[1]
    w = 0.1*np.random.randn(M)
    w_star = opt.fmin_bfgs(coss_entropy_error_fn, w, fprime=grad, args=(X, t, l,), disp=disp)
    return w_star

# opt_w_map = {}
# print("======================================")
# for p in range(-3,2):
#     l = 10 ** p
#     w_star = train(X_train_sub_norm, y_train, l)
#     opt_w_map[p] = w_star
#     print("λ:", l)
#     print("w⋆[:10] = ", w_star[:10])
#     print("======================================")


# ## (2 points) Evaluating the solution with the validation data
# So far, you have only used the training data. You now apply the learned model to the validation data and compare the prediction you get with the given validation labels.
#
# Report the performance measures
# * the number of false positives (FP),
# * the number of false negatives (FN),
# * the number of true positives (TP),
# * the number of true negatives (TN),
# * the error rate,
# * the specificity,
# * the sensitivity,
#
# for the different settings of $\lambda = 10^k, k=-3,-2,..,1$.

# In[9]:

def predict(X, w_star):
    """Using the learned parameter theta_best, predict on data Xtest"""
    X = add_ones(X)
    p = sig(np.dot(w_star, X.T))
    for i in range(len(p)):
        if p[i] > 0.5:
            p[i] = 1
        else:
            p[i] = 0
    return p

def confusion_matrix(prediction, labels):
    """Returns the confusion matrix for a list of predictions and (correct) labels"""
    assert len(prediction) == len(labels)
    def f(p, l):
        n = 0
        for i in range(len(prediction)):
            if prediction[i] == p and labels[i] == l:
                n += 1
        return n
    return np.matrix([[f(1, 1), f(1, 0)], [f(0, 1), f(0, 0)]])

def prediction_performance(X, t, w_star, report=True):
    y_predicted = predict(X, w_star)
    c_matrix = confusion_matrix(y_predicted, t)
    tp, fp, fn, tn = c_matrix.flatten().tolist()[0]

    # Rate at which an erroneous prediction occurs
    error_rate = (fp + fn) / (tp + fp + fn + tn)

    # AKA, True Negative Rate(TNR), is the proportion of people that tested negative and are negative
    specificity = tn / (fp + tn)

    # AKA, True Positive Rate(TPR) or recall, is the proportion of people that tested positive and are positive
    sensitivity = tp / (fn + tp)
    if report:
        print("False Positives:", tn)
        print("False Negatives:", fn)
        print("True Positives:", tp)
        print("True Negatives:", tn)
        print("Error Rate:", error_rate)
        print("Specificity:", specificity)
        print("Sensitivity:", sensitivity)
    else:
        return error_rate


# print("======================================")
# for p, w_star in opt_w_map.items():
#     print("λ:", 10**p)
#     prediction_performance(X_val_sub_norm, y_val, w_star)
#     print("======================================")


# ## (3 points) Finding an optimal regularisation constant
# We now consider the regularisation constant as a hyperparameter which we want to optimise.
# For this task we use the training and validation data together.
#
# Implement *s-fold cross-validation* with $s = 10$ to find an optimal regularisation constant which further reduces the error rates found in the previous question. Report the
# * optimal setting for the regularisation constant,
# * the first 10 components of the optimal parameter $\mathbf{w}^\star$, and
# * the same performance measures as specified in the previous question which you achieved with those settings.

# In[10]:

def s_fold_cross_validation(S, X, t, l):
    print("Starting S-fold cross-validation for λ =", l)
    X_groups = np.array_split(X, S)
    Y_groups = np.array_split(t, S)
    error_rates = np.ndarray(S)
    print("Training run:", end=' ')
    for i in range(S):
        print(i+1, end=' ')
        X_others = [X_groups[x] for x in range(S) if x!=i]
        Y_others = [Y_groups[x] for x in range(S) if x!=i]
        X_training = np.concatenate(tuple(X_others), axis=0)
        Y_training = np.concatenate(tuple(Y_others), axis=0)
        # np.random.seed(seed=int(i*l*1000))
        w_star = train(X_training, Y_training, l, disp=0)
        # y_predicted = predict(X_groups[i], w_star)
        # c_matrix = confusion_matrix(y_predicted, Y_groups[i])
        # tp, fp, fn, tn = c_matrix.flatten().tolist()[0]
        # error_rates[i] = (fp + fn) / (tp + fp + fn + tn)
        error_rates[i] = prediction_performance(X_groups[i], Y_groups[i], w_star, report=False)
    print("")
    return error_rates.mean()

def optimal_reg_hyperparameter(X, t):
    S = 10
    min_err_hp = (0, float("inf"))
    for p in range(-3,2):
        l = 10 ** p
        error_rate = s_fold_cross_validation(S, X, t, l)
        if error_rate < min_err_hp[1]:
            min_err_hp = (l, error_rate)
    return min_err_hp

# X_total = np.concatenate((X_train_sub_norm, X_val_sub_norm), axis=0)
# Y_total = np.concatenate((y_train, y_val), axis=0)

l, avg_err_rate = (10, 0.145)#optimal_reg_hyperparameter(X_total, Y_total)
# print("\nOptimal λ:", l)
# print("Avg. Error rate:", avg_err_rate, "\n")

# print("Training on training data with λ =", l)
# w_ml = train(X_train_sub_norm, y_train, l, disp=0)
# print("Optimal w⋆[:10] = ", w_ml[:10])
# print("\nPerformance measure with λ =", l)
# prediction_performance(X_val_sub_norm, y_val, w_star)


# ## (3 points) Feature selection from all 10000 features
#
# In this task, you will use all 10000 features of the input data.
#
# The goal is to find a subset of the 10000 features which improves the solution found so far.
#
# An improvement is when at least one of
# * the error, or
# * the size of your chosen subset
#
# decreases.
#
# Please explain your approach and the reasons why you have chosen it.
# Provide code to find the subset and report the number of features and the error you are able to achive.

# ### Solution

# In[30]:

def var_threshold(X, threshold=1.0):
        variances = np.var(X, axis=0)
        return np.asarray([var < threshold for var in variances])

# Apply variance threshold
mask = var_threshold(X_train)
X_train_ft = X_train[:,~mask]
X_val_ft   = X_val[:,~mask]

print(X_train_ft.shape)
print(X_val_ft.shape)

# Apply normalization
X_train_ft_mn = [ X_train_ft[:,col].mean() for col in np.arange(X_train_ft.shape[1])]
X_train_ft_sd = [ X_train_ft[:,col].std() for col in np.arange(X_train_ft.shape[1])]
X_train_ft = normalize(X_train_ft, X_train_ft_mn, X_train_ft_sd)
X_val_ft = normalize(X_val_ft, X_train_ft_mn, X_train_ft_sd)

# Remove nan and full zero features
mask = np.all(np.isnan(X_train_ft) | np.equal(X_train_ft, 0), axis=0)
X_train_ft = X_train_ft[:,~mask]
X_val_ft = X_val_ft[:,~mask]

print(X_train_ft.shape)
print(X_val_ft.shape)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
skobj = SelectKBest(f_classif, k=1000)
X_t_new = skobj.fit_transform(X_train_ft, y_train)
X_v_new = skobj.transform(X_val_ft)
print(X_t_new.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
lr = LogisticRegression(C=100, penalty="l1", dual=False).fit(X_t_new, y_train)
model = SelectFromModel(lr, prefit=True)
X_t_new = model.transform(X_t_new)
X_v_new = model.transform(X_v_new)
print(X_t_new.shape)

w_ml = train(X_t_new, y_train, l, disp=1)
print("Optimal w⋆[:10] = ", w_ml[:10])
print("\nPerformance measure with for subset:")
prediction_performance(X_v_new, y_val, w_ml)

# ## (2 points) Maximum likelihood and maximum a posteriori (MAP)
# We assume data samples $X_n = \{ x_1,\dots,x_n \}$ were generated i.i.d. from a uniform distribution with unknown positive parameter $\theta$:
# $$
#    \mathcal{U}(x \;|\; 0, \theta) =
# \begin{cases}
#  1/\theta & 0 \leq x \leq \theta \\
#  0        & \textrm{otherwise}   \\
# \end{cases}
# $$
#
# a) We now observe four data samples $ X_4 = \{ 6, 8, 9, 5\}$.
# Calculate $\theta_{ML}$, the maximum likelihood estimate of $\theta$ for the observed data.
#
# b) Calculate the posterior distribution of $\theta$
# given that the data $ X_4 = \{ 6, 8, 9, 5 \}$ have been observed. As prior for $\theta$
# use $p(\theta) = \mathcal{U}(x \;|\; 0, 10)$.
#
# c) Calculate $\theta_{MAP}$, the maximum a posteriori estimate of $\theta$ given the data $ X_4 $ and the prior $p(\theta)$ as in the previous question.
#
# Write down the calculations in $\LaTeX$.

# ### Solution

# ## (1 point) Variance of sum of random vartiables
# Prove that the following holds for the variance of a sum of two random variables
# $ X $ and $ Y $
# $$
# \operatorname{var}[X + Y] = \operatorname{var}[X] + \operatorname{var}[Y] + 2 \operatorname{cov}[X,Y],
# $$
# where $ \operatorname{cov}[X,Y] $ is the covariance between $X$ and $Y$.
#
# For each step in your proof, provide a verbal explanation why this transformation step holds.

# ### Solution

# ## (1 point) Matrix-vector identity proof
# Given a nonsingular matrix $ \mathbf{A} $ and a vector $ \mathbf{v} $ of comparable
# dimension, prove the following identity:
# $$
#  (\mathbf{A} + \mathbf{v} \mathbf{v}^T)^{-1}
#    = \mathbf{A}^{-1} - \frac{(\mathbf{A}^{-1} \mathbf{v}) (\mathbf{v}^T \mathbf{A}^{-1})}
#                        {1 + \mathbf{v}^T \mathbf{A}^{-1} \mathbf{v}}.
# $$

# ### Solution

# ## (3 points) Change of variance
# In Bayesian Linear Regression, the predictive distribution
# with a simplified prior
#   $ p(\mathbf{w}  \;|\;  \alpha) = \mathcal{N}(\mathbf{w} \;|\; \mathbf{0}, \alpha^{-1}\mathbf{I}) $
# is a Gaussian distribution,
# $$
# p(t  \;|\;  \mathbf{x}, \mathbf{t}, \alpha, \beta)
# = \mathcal{N} (t \;|\; \mathbf{m}_N^T \boldsymbol{\mathsf{\phi}}(\mathbf{x}), \sigma_N^2(\mathbf{x}))
# $$
# with variance
# $$
#   \sigma_N^2(\mathbf{x}) = \frac{1}{\beta} + \boldsymbol{\mathsf{\phi}}(\mathbf{x})^T \mathbf{S}_N \boldsymbol{\mathsf{\phi}}(\mathbf{x}).
# $$
#
# After using another training pair $ \left( \mathbf{x}_{N+1}, t_{N+1} \right) $ to adapt ($=$learn) the model,
# the variance of the predictive distribution becomes
#
# $$
#   \sigma_{N+1}^2(\mathbf{x}) = \frac{1}{\beta} + \boldsymbol{\mathsf{\phi}}(\mathbf{x})^T \mathbf{S}_{N+1} \boldsymbol{\mathsf{\phi}}(\mathbf{x}).
# $$
#
# a) Define the dimensions of the variables.
#
# b) Prove that the uncertainties $ \sigma_N^2(\mathbf{x}) $ and
# $ \sigma_{N+1}^2(\mathbf{x}) $ associated with the
# predictive distributions satisfy
#
# $$
#   \sigma_{N+1}^2(\mathbf{x}) \le \sigma_N^2(\mathbf{x}).
# $$
# *Hint: Use the Matrix-vector identity proved in the previous question.*
#
# c) Explain the meaning of this inequality.
#
#

# ### Solution
