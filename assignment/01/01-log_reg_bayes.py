
# coding: utf-8

# # Classification with Logistic Regression, Bayes Linear Regression

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

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import scipy.optimize as opt
import scipy.stats as stats

get_ipython().magic('matplotlib inline')


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

# In[2]:

X_train = np.loadtxt("arcene_train.data")
y_train = np.loadtxt("arcene_train.labels")
X_val   = np.loadtxt("arcene_valid.data")
y_val   = np.loadtxt("arcene_valid.labels")

print(X_train.shape)
print(X_val.shape)


# As the 10000-dimensional input space might lead to long computation times, we have prepared a subset of 220 features. Unless otherwise stated, you will from now on work only with these subset of the training and validation data.

# In[3]:

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

# Converting -1 class to 0
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

# In[164]:

def get_mean_sd(X):
    """Returns the mean and standard deviation of each column of the given data-set"""
    X_mean = [ X[:,col].mean() for col in np.arange(X.shape[1])]
    X_sd = [ X[:,col].std() for col in np.arange(X.shape[1])]
    return (X_mean, X_sd)

def normalize(X, mu, sd):
    """ Normalizes each column of the given data-set such that 
        they have zero mean and unit-variance
    """
    return (X - mu)/sd

# Get the column-wise mean, sd of the training data.
X_train_sub_mean, X_train_sub_sd = get_mean_sd(X_train_sub)

# Reporting the parameters (mean, sd) for the first 10 dimensiond of input data
print("Mean for first 10 dimension:", X_train_sub_mean[:10])
print("\nStandard Deviation for first 10 dimension:", X_train_sub_sd[:10])

# Normalizing the training data and validation data using the mean, sd of the training data.
# The reason why we use the training set (mean, sd) is to simulate the effect of the selected
# model on an un-seen data.
X_train_sub_norm = normalize(X_train_sub, X_train_sub_mean, X_train_sub_sd)
X_val_sub_norm = normalize(X_val_sub, X_train_sub_mean, X_train_sub_sd)


# ### Solution
# 
# ##### Circumstances where normalization is an Advantage
# * When features are represented in different scale. Since the error function uses an euclidean distance to calculate error in prediction, the features with larger value (due to it's scale) get more influence on prediction when compared to features whose numerical values are small
# * When there are outliers in the data
# * When we know (assume) that the data has a normal distribution
# * When we know we are going to use optimization algorithms like gradient discent which converge much faster when the data is normalized.
# 
# ##### Circumstances where normalization is Counter-productive
# * When the input distribution is known to be **not** gaussian
# * When all the features are in the same range, there by rendering normalization useless
# * When some features are known to influence more on the prediction of the target, normalization is counter-productive.

# ## (1 point) Error function for logistic regression with quadratic regularisation
# Define a Python function calculating the *cross-entropy* $ E(\mathbf{w}) = - \ln p(\mathbf{t} \;|\; \mathbf{w}) $for logistic regression with two classes and quadratic (l2) regularisation for a given parameter vector $\mathbf{w}$.

# In[166]:

def sig(X):
    """S shaped function, known as the sigmoid"""
    return 1 / (1 + np.exp(- X))

def coss_entropy_error_fn(w, X, t, l):
    """Return the cross-entropy value for a given set of weight vector, w"""
    
    # Calculating y = Sigma(a)
    p_1 = sig(np.dot(w, X.T))
    
    # Calculating the two parts of the summation as dot-products.
    erfn_1 = np.dot(t, np.log(p_1))
    erfn_2 = np.dot((1-t), np.log(1 - p_1))
    
    # Evaluation of cost function value with an L2 regularization parameter.
    cost = -1 * (erfn_1 + erfn_2) + (l/2) * np.dot(w.T, w)
    
    # Return a single value, cost is 1x1 array.
    return cost.mean()


# ## (1 point) Gradient for logistic regression with quadratic regularisation
# Define a Python function calculating the gradient of the above *cross-entropy* for logistic regression with two classes and quadratic (l2) regularisation.

# In[169]:

def grad(w, X, t, l):
    """Returns the gradient vector for a given set of weight vector, w"""
    
    # Calculating y = Sigma(a)
    p_1 = sig(np.dot(w, X.T))
    
    # The gradient is calculated as (error in estimation)*(input) + (reg)(weights)
    err = p_1 - t
    grd = np.dot(err, X) + l * w    
    return grd


# ## (1 point) Finding the optimal parameters for given regularisation
# Using the error function and the gradient defined above, you now setup the optimisation finding the optimal parameter vector $\mathbf{w}^\star$ for the training data and a fixed regularisation constant $\lambda $.
# Use the function *scipy.optimize.fmin_bfgs* as optimiser.
# 
# For each $\lambda= 10^k, k=-3,-2,..,1$, report the first 10 components of the optimal parameter $\mathbf{w}^\star$ found.

# In[171]:

def add_ones(X):
    """Add a column of ones to the end of a given data set"""
    N = X.shape[0]
    D = X.shape[1]
    M = D + 1
    tmp = np.ones((N,M))
    tmp[:,:-1] = X
    return tmp

def train(X, t, l, disp=1):
    """ Returns an optimal weight vector after performing Logistic Regression 
        on the given input data set.
    """
    
    # Add a bias term so that the decision hyperplane has the freedom of translation.
    X = add_ones(X)
    M = X.shape[1]
    
    # Randomize sarting weights to a small value
    w = 0.1*np.random.randn(M)
    
    # Find optimal weights that minimizes the cost function.
    w_star = opt.fmin_bfgs(coss_entropy_error_fn, w, fprime=grad, args=(X, t, l,), disp=disp)
    return w_star

opt_w_map = {}
print("======================================")
for p in range(-3,2):
    l = 10 ** p
    w_star = train(X_train_sub_norm, y_train, l)
    opt_w_map[p] = w_star
    print("λ:", l)
    print("w⋆[:10] = ", w_star[:10])
    print("======================================")


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

# In[172]:

def predict(X, w_star):
    """Using the learned parameter w_star, predict on validation dataset"""
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
    """Reports the predictive performance of the learned weights or returns the error rate"""
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
        print("False Positives:", fp)
        print("False Negatives:", fn)
        print("True Positives:", tp)
        print("True Negatives:", tn)
        print("Error Rate:", error_rate)
        print("Specificity:", specificity)
        print("Sensitivity:", sensitivity)
    else:
        return error_rate
    
        
print("======================================")
for p, w_star in opt_w_map.items():
    print("λ:", 10**p)
    prediction_performance(X_val_sub_norm, y_val, w_star)
    print("======================================")


# ## (3 points) Finding an optimal regularisation constant
# We now consider the regularisation constant as a hyperparameter which we want to optimise.
# For this task we use the training and validation data together.
# 
# Implement *s-fold cross-validation* with $s = 10$ to find an optimal regularisation constant which further reduces the error rates found in the previous question. Report the 
# * optimal setting for the regularisation constant,
# * the first 10 components of the optimal parameter $\mathbf{w}^\star$, and 
# * the same performance measures as specified in the previous question which you achieved with those settings.

# In[240]:

def cross_validate(S, X, t, disp=True):
    """ Return the most Optimal value for the regularization hyper-parameter by 
        performing S-fold cross validation for all allowed values of λ."""    
    X_groups = np.array_split(X, S)
    Y_groups = np.array_split(t, S)    
    min_err_hp = (0, float("inf"))
    for p in range(-3,2):        
        l = 10 ** p
        if disp: print("Starting S-fold cross-validation for λ =", l)
        if disp: print("Training run:", end=' ')
        # i represents the held-out group
        error_rates = np.ndarray(S)
        for i in range(S):
            if disp: print(i+1, end=' ')
            X_others = [X_groups[x] for x in range(S) if x!=i]
            Y_others = [Y_groups[x] for x in range(S) if x!=i]
            X_training = np.concatenate(tuple(X_others), axis=0)
            Y_training = np.concatenate(tuple(Y_others), axis=0)
            w_star = train(X_training, Y_training, l, disp=0)

            # Prediction on the help-out group
            error_rates[i] = prediction_performance(X_groups[i], Y_groups[i], w_star, report=False)
        if disp: print("")
        error_rate = error_rates.mean()
        
        if error_rate < min_err_hp[1]:
            min_err_hp = (l, error_rate)
    return min_err_hp

# Use the whole data to find the optimal value for λ
X_total = np.concatenate((X_train_sub_norm, X_val_sub_norm), axis=0)
Y_total = np.concatenate((y_train, y_val), axis=0)

l, avg_err_rate = cross_validate(10, X_total, Y_total)

print("\nOptimal λ:", l)
print("Avg. Error rate:", avg_err_rate, "\n")

# Train and predict with the optimal regularization parameter.
print("Training on training data with λ =", l)
w_ml = train(X_train_sub_norm, y_train, l, disp=0)
print("Optimal w⋆[:10] = ", w_ml[:10])
print("\nPerformance measure with λ =", l)
prediction_performance(X_val_sub_norm, y_val, w_ml)


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

# ## Solution
# 
# ### Subset Selection Process
# 
# Step 1: Use Variance Threshold filtering technique, weed out those feature having near zero variance, i.e. those feature that vary very little in the sample space.
# 
# Step 2: Remove those fetures that have near zero (range: [-0.1,0.1]) correlation with the output. This means that as the value of the feature changes there is little to no change in the output. We use the Pearson product-moment correlation coefficient (`scipy.stats.pearsonr`) to get the correlation score for a feature.
# 
# Step 3: Run a Greedy forward feature selection technique which ranks the features using the fisher discriminant score and calculates the optimal feature subset size using cross-validation (Filter + Wrapper feature selection technique). 
# 
# * The fisher discriminat score is given as the ratio of between class variance to the within class variance. The higher the score better the feature. 
# * Since this is a computationally intesive process, we start the checking from a feature subset size of 100, terminating the check at 220.
# * Optimal size of the feature subset using Fisher discriminant scoring is calculated to be **205**.
# * Since it takes around 10min the run the cross-validation algorithm, the call to the function **`get_optimal_k`** is commented out.
# 
# Step 4: Using Fisher discriminant scoring obtain a final mask using subset size equal to 205.
# 
# Step 5: With the final feature subset, perform logistic regression and report the performance on the validation set.
# 
# ##### References
# * https://en.wikipedia.org/wiki/Feature_selection
# * http://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance
# * http://www.maxwellsci.com/print/rjaset/v7-625-638.pdf
# * https://en.wikipedia.org/wiki/Linear_discriminant_analysis

# In[242]:

def var_threshold(X, threshold=0):
    """ Returns a mask of all features falling below the given variance threshold."""
    variances = np.var(X, axis=0)
    if threshold==0:
        return np.asarray([np.isclose([var],[threshold])[0] for var in variances])
    else:
        return np.asarray([var < threshold for var in variances])
    
def remove_zero_correlation(X, y, atol=10**-8):
    """ Return a mask of all features that have their pearsonr correaltion 
        coefficient within a range around zero.
    """
    feature_score = []
    for col in range(X.shape[1]):
        f, _ = stats.pearsonr(X[:,col], y)
        feature_score.append(np.isclose([f], [0], atol=atol)[0])
    return np.asarray(feature_score)

def fisher_discriminant(x, y):
    """Return the Fisher discriminant score: ratio of between class variance to within class variance."""
    return ((x.mean() - y.mean())**2)/(x.var() + y.var())
    
def univariate_fs(X, y, k=275):
    """ Returns a mask of the first k features ranked by the Fisher discriminant score."""
    feature_score = {}
    import operator
    for col in range(X.shape[1]):
        f = fisher_discriminant(X[:,col], y)
        feature_score[col] = f
    feature_score = sorted(feature_score.items(), key=operator.itemgetter(1), reverse=True)[:k]
    return np.asarray([col for col, _ in feature_score])


def cross_validate_k(S, X, t, l, min_feature=100, max_feature=250,disp=True):
    """Performs Cross validation to optimize the number of features that gives the best error rate."""
    X_groups = np.array_split(X, S)
    Y_groups = np.array_split(t, S)    
    min_err_hp = (0, float("inf"))
    for k in range(min_feature, max_feature+1):
        if disp: print("Starting S-fold cross-validation for k =", k)
        if disp: print("Training run:", end=' ')
        # i represents the held-out group
        error_rates = np.ndarray(S)
        for i in range(S):
            if disp: print(i+1, end=' ')
            X_others = [X_groups[x] for x in range(S) if x!=i]
            Y_others = [Y_groups[x] for x in range(S) if x!=i]
            X_training = np.concatenate(tuple(X_others), axis=0)
            Y_training = np.concatenate(tuple(Y_others), axis=0)
            
            # Feature selection has to be done for each partition to generalize the fitting.
            # It leads to over-fitting to the validation data if the scoring & selection is done 
            # once on the entire dataset and then masked locally in a partition.
            mask = univariate_fs(X_training, Y_training, k=k)
            X_training_subset = X_training[:, mask]
            X_val_subset = X_groups[i][:, mask]
            
            w_star = train(X_training_subset, Y_training, l, disp=0)
            # Prediction on the hold-out partition
            error_rates[i] = prediction_performance(X_val_subset, Y_groups[i], w_star, report=False)        
        error_rate = error_rates.mean()
        if disp: print("; Error rate:",error_rate)
        if error_rate < min_err_hp[1]:
            min_err_hp = (k, error_rate)
    return min_err_hp

def get_optimal_k(X, t, l, min_feature=100, max_feature=250, disp=False):
    """Initiates cross validation and return the optimal value of k"""
    print("\nStarting Greedy Forward Checking Selection Algorithm.")
    print("Performing 5-fold cross-validation with Fisher discriminant scoring.")
    print("This takes about 10 min... grab a cup of coffee :)")
    print("Minimum Feature count:", min_feature)
    print("Forward Checking Threshold:", max_feature)

    import time
    start = time.clock()
    k, _ = cross_validate_k(5, X, t, l, min_feature, max_feature, disp=disp)
    stop = time.clock() - start
    print("Time to find optimal k:", stop, "sec")
    return k

X_total = np.concatenate((X_train, X_val), axis=0)
Y_total = np.concatenate((y_train, y_val), axis=0)

# Apply variance threshold
mask = var_threshold(X_total, threshold=0)
X_total    = X_total[:, ~mask]

print("Removed zero-variance features: ", X_total.shape[1])

# Apply normalization
X_total_mn, X_total_sd = get_mean_sd(X_total)
X_total = normalize(X_total, X_total_mn, X_total_sd)

# remove near zero correlation
mask = remove_zero_correlation(X_total, Y_total, atol=10**-1)
X_total = X_total[:, ~mask]

print("Removed features with pearson correlation score in range [-0.1,0.1]: ", X_total.shape[1])

# Get Optmial value of k
# k = get_optimal_k(X_total, Y_total, l, min_feature=100, max_feature=220, disp=False)

# The k value of 205 was found using get_optimal_k function 
# commented out above (its output is shown below.) The function takes close to 10 min to execute.

# Take first 205 features ranked by the Fisher discriminant score.
mask = univariate_fs(X_total, Y_total, k=205)
X_total = X_total[:, mask]

print("\nSelected top 205 features, ranked by the Fisher discriminant score:", X_total.shape[1])

# Split total to training and validation set (the initial set order is preserved)
X_train_new_sub_norm, X_val_new_sub_norm = tuple(np.vsplit(X_total, 2))

w_ml = train(X_train_new_sub_norm, y_train, l)
print("Optimal w⋆[:10] = ", w_ml[:10])
print("\nPerformance measure with subset:", X_train_new_sub_norm.shape)
prediction_performance(X_val_new_sub_norm, y_val, w_ml)


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
