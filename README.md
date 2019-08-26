# Naive-Bayes-from-Scratch-with-R

Naive Bayes is one of the most used models for classification problems. If this theorem is understood, then most of the other algorithms can be understood easily.
It’s name comes from Bayes Theorem that is covered in Probability Courses.

P(c|x)= (P(c)*P(x|c))/ P(x)
P(c|x) is the posterior probability of class (c, that will be predicted) given predictor (x, other attributes).
P(c) is the prior probability of class. (For train Data)
P(x|c) is the likelihood which is the probability of predictor given class (The values will obtained from train data is used as parameters for test data)
P(x) is the prior probability of predictor

The reason for calling it Naive Bayes is that two of the assumptions of Bayes Theorem are not (strictly) valid for this model.
Firstly, by this approach we assume that there is a strong independence between features. For example:for our data set it means Sepal.Length and Sepal.Width (from Figure 2) are independent for any observation and the same for other attributes.
Secondly, we assume that the features are normally distributed. In the below plot, the distribution of train data for Sepal.Length can be observed for 3 Classes. And also the density plot is shown. As can be seen the data is far from being normally distributed. But at the end classification is very successful.

For this task, famous (Fisher's or Anderson's) iris data set is used. Below the observations of data set can be seen. In Species we have 3 classes: Setosa, Virginica and versicolor.	
  Sepal.Length Sepal.Width Petal.Length Petal.Width Species
1          5.1         3.5          1.4         0.2  setosa
2          4.9         3.0          1.4         0.2  setosa
3          4.7         3.2          1.3         0.2  setosa
4          4.6         3.1          1.5         0.2  setosa
5          5.0         3.6          1.4         0.2  setosa
6          5.4         3.9          1.7         0.4  setosa

When we applied the naive Bayes Theorem for this data set. We want to classify Species (the last column- 3 classes). It means the other 4 columns are independent from each other and these 4 variables are normally distributed.
Data set is divided  2 randomly subsets. The first one is train data set and the other is test data set. Our model is constructed on the train data. Then by test data, the accuracy of model is tested. Understanding the intuition and logic of Naive Bayesian model helps to grasp other models and applications also. Obviously without the computing power of computers, these calculations would be very difficult.
