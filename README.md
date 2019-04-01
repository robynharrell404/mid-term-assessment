# McKesson Data Science Midterm Examination


### Q-01

*10 PTS*

Due to Frank's bad rating as a passenger in an Uber, not many cars will pick him up when hailed. Frank has figured out that he has an 72% chance of not being picked up when he hails one of their cars. Out of the next 14 cars that Frank tries to hail, what is the probability that at least 10 will pick him up. Plot a bar chart that best illustrates the overall experiment.

binomial dist

s = scs.binom(14, 0.72)

s.cdf(10)

### Q-02

*5 PTS*

You fit a linear regression to predict SAT score with many predictors, one of which is whether or not the student was homeschooled. Beta_homeschool = -40. How do you interpret the coefficient?

the predictor beta_homeschool has an inverse relationship with the outcome, whether or not a student was homeschooled. would need to look at the z score to see if this beta coefficient is statistically different than 0.

### Q-03

*10 PTS*

Give an example of a confusion matrix with accuracy > 90%, but both precision < 5% and recall < 5%. Prove your answer.

TP=3
TN=270
FP=0
FN=27
Accuracy = (TP + TN) / (TP+TN+FP+FN)
Accuracy -> 90

Precision=TP/(TP+FP)
Precision -> 0

Recall = TP / (TP + FN)
Recall -> 0

### Q-04

*10 PTS*

The `playgolf.csv` file has data on when is the best time to play golf. Imagine you're constructing a decision tree and the first split you make is on the `Outlook` column with the value `Sunny`. How much information have you gained by making that split?

#calcualte entropy of the sample
A=4/14
B=5/14
C=5/14

A_ent = (A * math.log(A, 2)) * -1
B_ent = (B * math.log(B, 2)) * -1
C_ent = (C * math.log(C, 2)) * -1
ent_prnt = A_ent + B_ent + C_ent
ent_prnt=1.5774062828523452

a = 2/5
b = 3/5

#calcualte entropy of left child, where entropy = proportion of class in the sample times to the log base2 of the proportion of class in the sample
ent_lc = ((a * (math.log(a,2))) + (b * (math.log(b, 2))))*-1
ent_lc=0.9709505944546686

c = 7/9
d = 2/9

#calcualte entropy of left child, where entropy = proportion of class in the sample times to the log base2 of the proportion of class in the sample
ent_rc = ((c * (math.log(c,2))) + (d * (math.log(d, 2))))*-1
ent_rc=0.7642045065086203

IG = ent_prnt - ((5/14) * ent_lc) + ((9/14) * ent_rc)*-1
IG = 0.7393638877915648



### Q-05

*5 PTS*

You build a Decision Tree and get these stats:

```
train set accuracy:  90%
train set precision: 92%
train set recall:    87%

test set accuracy:   60%
test set precision:  65%
test set recall:     52%
```

What's going on? What tactic(s) do we have to modify our Decision Tree to fix the issue?

The model seems to perform well on the training data; however, not so well on the test data.  Could be the result of overfitting.  Try some pruning.

Pre-pruning ideas (pruning while you build the tree)
○ Leaf size: stop splitting when the number of samples left gets small enough
○ Depth: stop splitting at a certain depth (after a certain number of splits)
○ Purity: stop splitting if enough of the examples are the same class
○ Gain threshold: stop splitting when the information gain becomes too small

Post-pruning ideas (pruning after you’ve finished building the tree)
○ Merge terminal nodes (i.e., undo your split) if doing so decreases error in your test set
○ Set the maximum number of terminal nodes; this is a form of regularization

Prune by varying the following hyperparameters: max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes

### Q-06

*5 PTS*

You have a linear regression model that is overfitting on training data. What regularization techniques do you have to help with this issue? How do these techniques differ?

L1 (Lasso) minimizes number of parameters (feature engineering) and L2 (ridge regression) minimizes size of parameters

### Q-07

*10 PTS*

Implement the function `column_averages`. It takes the name of a csv file and returns a dictionary whose keys are the names of the columns and values are the mean of the column.

The file `example.csv` has been provided. The code should run as follows.

```
column_averages('data/example.csv')
{'A': 0.35999999999999999, 'C': 64.200000000000003, 'B': 1330.8}
```

### Q-08

*10 PTS*

I have two 6-sided dice. One is a standard die with the numbers 1-6. The other has three 3's and three 6's. You grab one at random and roll a six. What is the probability that you chose the fair die?

fair die prob is 1/6 and unfair die prob is 3/6

need to use bayes model to determine prob

### Q-09

*5 PTS*

You are given the following data of student grades in a 5-by-5 matrix.

|  id | Assignment 1 score | Assignment 2 score | Total score | Score percentage |
| --- | ------------------ | ------------------ | ----------- | ---------------- |
|   1 |                 12 |                 19 |          31 |             0.62 |
|   2 |                 20 |                 21 |          41 |             0.82 |
|   3 |                 15 |                 22 |          37 |             0.74 |
|   4 |                 14 |                  9 |          23 |             0.46 |
|   5 |                 25 |                 25 |          50 |             1.00 |

Make a guess of the rank of the matrix. Explain your guess.

rank -> number of linearly independent rows or columns is called rank of matrix.
c3=c1+c2
c4=(c1+c2+c3)/100
There are two linearly independent columns (c1 and c2) hence its rank is 2.

### Q-10

*5 PTS*

You have a model that predicts the price of the house given the number of square feet, number of rooms, age and size of plot. You use a regression model to predict the price. You try varying the degree of the polynomial used to fit the data. This graph shows mean squared error on the training and testing sets based on the "complexity" of the model (degree of the polynomial).

What is going on in this graph? What would be an optimal choice for the degree of the polynomial?

quadratic

![House](housing.png)

### Q-11

*5 PTS*

You have three models which classify emails as spam or not spam. You have tested it on a dataset of size 1000, with 100 spam emails and 900 non-spam emails. The accuracy of your models on that dataset is as follows:

```
 Model A: 83%
 Model B: 88%
 Model C: 79%
```

Create a model which would have higher accuracy than any of the three models.

### Q-12

*5 PTS*

Logistic Regression internally creates a linear equation. How does the model use this linear equation to produce probabilities?
logistic regression estimates the log odds of an event with the sigmoid function

### Q-13

*5 PTS*

What is grid search doing when it's running?

grid search is best for ridge regression, i.e. reducing the size of predictors, runs k fold validation on a data set so that the model is exposed to all of the data and looks for optimal lambda, where lambda minimizes the cost function

### Q-14

*5 PTS*

If your linear regression model had a negative R-squared, what would be the simplest thing you could do to improve the model?
Only use the mean value as your predictor

### Q-15

*5 PTS*

Due to overfitting on your linear regression model, you decide to use L2 regularization. How do you know when lambda is tuned to the correct value for optimal model performance? when the cost function is minimized

### Q-16

*5 PTS*

When scaling your data, why do you scale the test data set from the values in the training data set?
so that the scaling is appropriate and you are comparing apples to apples

### Q-17

*5 PTS*

When computing probabilities, the CDF of a binomial distribution uses sigma notation while the CDF of a gamma distribution uses integral notation. Why is this? because the binomial distribution is discrete and the gamma function is continuous

### Q-18

*5 PTS*

Suppose you compare a five-predictor model with a higher R-squared to a one-predictor model. Is the five-predictor model better? Why?  Not necessarily.  The 5 predictor model will have higher r squared simply due to the fact of having 5 predictors compared to 1 predictor.

### Q-19

*5 PTS*

Why should you use relatively low dimensional vectors when using a KNN model?

### Q-20

*10 PTS*

At the manufacturing plant you work at, one empty 12-ounce aluminum soda should weigh about 15 grams. But you think the equipment has malfunctioned and is altering the weight of the cans. You perform a hypothesis test with the following data of measured can weight and a 97% confidence interval. What are your conclusions?

```
14.76515145, 15.66921114, 20.15497514, 14.25750049, 13.03577315,
8.26272148, 19.20985755, 12.04375879, 10.91085244, 11.68537558,
15.76500706,  9.10184183, 12.27624773, 12.54001656, 17.38484272,
17.57900201, 14.65171242, 15.81325345, 12.16072462,  9.66999714,
15.39782075, 14.71051375, 12.87376932, 13.16569868, 13.1352791 ,
17.62503595, 15.85801613, 16.08111466, 12.74932095,  8.78455273,
15.67603836, 20.85680148, 13.55898338, 16.12441324, 18.89939279,
9.74224957, 12.93047653, 15.26948315, 18.85026598,  9.6622572 ,
14.69180529,  7.98256903, 11.98738448, 13.09929852, 18.00647745,
13.37693129, 11.63440091, 19.55690723, 11.32272672,  9.65382463
```
S=[14.76515145, 15.66921114, 20.15497514, 14.25750049, 13.03577315,
8.26272148, 19.20985755, 12.04375879, 10.91085244, 11.68537558,
15.76500706,  9.10184183, 12.27624773, 12.54001656, 17.38484272,
17.57900201, 14.65171242, 15.81325345, 12.16072462,  9.66999714,
15.39782075, 14.71051375, 12.87376932, 13.16569868, 13.1352791 ,
17.62503595, 15.85801613, 16.08111466, 12.74932095,  8.78455273,
15.67603836, 20.85680148, 13.55898338, 16.12441324, 18.89939279,
9.74224957, 12.93047653, 15.26948315, 18.85026598,  9.6622572 ,
14.69180529,  7.98256903, 11.98738448, 13.09929852, 18.00647745,
13.37693129, 11.63440091, 19.55690723, 11.32272672,  9.65382463]

np.mean(S)

np.var(S)

h0=15
#h1 != 15
s=50
sd=10.41**0.5
xbar=14.00
alpha=0.025

se = sd/(s**0.5)
z = (xbar - h0) / se
se, z

norm = scs.norm(0, 1)
smple = scs.norm(h0, se)

1-norm.cdf(z)

do not reject null hypothesis
