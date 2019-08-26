---
title: "Learning Activity: Naive Bayes by R"
author: "Yasin Cibuk"
date: "February 13, 2019"
data : "R data set"
description-of-dataset: "This famous (Fisher's or Anderson's) iris data set gives the measurements
in centimeters of the variables sepal length and width and petal length and width, respectively,
for 50 flowers from each of 3 species of iris. The species are Iris setosa, versicolor, and virginica. "
---
#In machine learning, naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem 
  #with strong (naive) independence assumptions between the features.--https://en.wikipedia.org/wiki/Naive_Bayes_classifier

#Removing old variables
rm(list = ls())              # Remove everything from your current workspace
ls()                         # Anything there? Nope.
getwd()
#Setting working directory
setwd("~/Desktop/~~~")
getwd()

#Install packages and load necessary libraries
library(datasets)
library(data.table)
library(dplyr)
library(caTools)
library(plyr)

#Loading Dataset
data(iris)
summary(iris)
df<-iris # clone data set

#Have a look at the data set
head(df)
str(df)
summary(df)
##We have 5 variables. 4 of them are numerical and one (Species) is categorical. And 150 observations.
## In Species we have three Categories (or Classes).
#Split Data as Train and Test
set.seed(101) 
sample = sample.split(df$Species, SplitRatio = .75)
train = subset(df, sample == TRUE)
test  = subset(df, sample == FALSE)

##Abstractly, naive Bayes is a conditional probability model: posterior = (prior*likelihood)/evidence
#https://www.saedsayad.com/naive_bayesian.htm
#https://www.machinelearningplus.com/predictive-modeling/how-naive-bayes-algorithm-works-with-example-and-full-code/
#P(c|x) is the posterior probability of class (c, target) given predictor (x, attributes).
#P(c) is the prior probability of class.
#P(x|c) is the likelihood which is the probability of predictor given class.
#P(x) is the prior probability of predictor.
## P(c|x)= (P(c)*P(x|c))/ P(x)

#For our example lets formulate the probabilities
# P(Species is "Setosa"| Species other Attributes)= (P(Species is "Setosa")*P(Species other Attributes|Species is "Setosa"))/ P(Species other Attributes)
# P(Species is "Versicolor"|Species other Attributes)= (P(Species is "Versicolor")*P(Species other Attributes|Species is "Versicolor"))/ P(Species other Attributes)
# P(Species is "Virginica"|Species other Attributes)= (P(Species is "Virginica")*P(Species other Attributes|Species is "Virginica"))/ P(Species other Attributes)

#Posterior("Setosa")= (P("Setosa")*P("Sepal.Length"|"Setosa")*P("Sepal.Width"|"Setosa")*P("Petal.Length"|"Setosa")*P("Petal.Width"|"Setosa"))/ Marginal Probability
#Posterior("Virginica")= (P("Virginica")*P("Sepal.Length"|"Virginica")*P("Sepal.Width"|"Virginica")*P("Petal.Length"|"Virginica")*P("Petal.Width"|"Virginica"))/ Marginal Probability
#Posterior("Versicolor")= (P("Versicolor")*P("Sepal.Length"|"Versicolor")*P("Sepal.Width"|"Versicolor")*P("Petal.Length"|"Versicolor")*P("Petal.Width"|"Versicolor"))/ Marginal Probability

#Now lets calculate P("Setosa"), P("Virginica") and P("Versicolor") in our train data set
p_setosa = length(which(train$Species == "setosa"))/length(train$Species)
p_versicolor = length(which(train$Species == "versicolor"))/length(train$Species)
p_virginica = length(which(train$Species == "virginica"))/length(train$Species)

#P("Sepal.Length"|"Setosa")*P("Sepal.Width"|"Setosa")*P("Petal.Length"|"Setosa") are the likelihood. From the likelihood
#it seems each feature is independent. But this is not true. That is why it is called "Naive Bayes"
#Also we assume that all of these features are normally distributed. That is to say P("Sepal.Length"|"Setosa") is calculated
#by probability density function of normal distribution--Confused about Likelihood and Probability?--> Then :https://www.youtube.com/watch?v=pYxNSUDSFH4

#Checking Histograms whether attributes of Species are really normally distributed??
#As can be observed from the previous histograms for Sepal.Length. It is only an assumption.
### Sepal.Length histograms and densities by groups###########
mu <- ddply(train, "Species", summarise, grp.mean=mean(Sepal.Length))
ggplot(train, aes(x=Sepal.Length, color=Species)) +
  geom_histogram(aes(y=..density..), position="identity", alpha=0.5)+
  geom_density(alpha=0.6)+
  geom_vline(data=mu, aes(xintercept=grp.mean, color=Species),
  linetype="dashed")+
  scale_color_manual(values=c("#999999", "#E69F00", "#56B4E9"))+
  scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9"))+
  labs(title="Sepal.Length histogram plot",x="Sepal.Length", y = "Density")+
  theme_classic()

#Another Visualization of Histogram
p<-ggplot(train, aes(x=Sepal.Length, color=Species)) +
  geom_histogram(fill="white", position="dodge")+geom_density(alpha=0.6)+
  scale_color_manual(values=c("#999999", "#E69F00", "#56B4E9"))+
  scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9"))+
  labs(title="Sepal.Length histogram plot",x="Sepal.Length", y = "Density")+
  theme_classic()
  geom_vline(data=mu, aes(xintercept=grp.mean, color=Species),
             linetype="dashed")
# Discrete colors
p + scale_color_brewer(palette="Dark2") +
  theme_minimal()+theme_classic()+theme(legend.position="top")

#Below I calculate mean and variance of eache species for every other variable of train data.
#If we need to calculate the likelihood we will use these values.
table<-train %>% group_by(Species) %>% summarise_each(funs(mean, sd))


#The below function is the density function of normal distribution. If it is called
# it will give the likelihood of value for given mean and variance.
likelihood<- function(x, mean_y, variance_y){
  # Input the arguments into a probability density function
  p = 1/(sqrt(2*pi*variance_y)) * exp((-(x-mean_y)**2)/(2*variance_y))
  return(p)
}

#Since denominator is the same for three posterior probability. I do not take into consiretaion denominator.
#By comparison (result of posterior1,posterior2,posterior3), the highest probability shows the group of sample_data
posterior1 <-function(x,priorprob){
  #We are calculating posterior for Setosa (denominator is ignored)
  p_Sepal.Length1 <- likelihood(x$Sepal.Length,table[1,2],table[1,6])
  p_Sepal.Width1 <- likelihood(x$Sepal.Width,table[1,3] , table[1,7])
  p_Petal.Length1 <- likelihood(x$Petal.Length,table[1,4] , table[1,8])
  p_Petal.Width1 <- likelihood(x$Petal.Width,table[1,5], table[1,9])
  return(p1=priorprob*p_Sepal.Length1*p_Sepal.Width1*p_Petal.Length1*p_Petal.Width1)
}

posterior2 <-function(x,priorprob){
  #We are calculating posterior for Versicolor (denominator is ignored)
  p_Sepal.Length2 <- likelihood(x$Sepal.Length,table[2,2] ,table[2,6])
  p_Sepal.Width2 <- likelihood(x$Sepal.Width,table[2,3] , table[2,7])
  p_Petal.Length2 <- likelihood(x$Petal.Length,table[2,4] , table[2,8])
  p_Petal.Width2 <- likelihood(x$Petal.Width,table[2,5], table[2,9])
  return(p2=priorprob*p_Sepal.Length2*p_Sepal.Width2*p_Petal.Length2*p_Petal.Width2)
}

posterior3 <-function(x,priorprob){
  ##We are calculating posterior for Virginica (denominator is ignored)
  p_Sepal.Length3 <- likelihood(x$Sepal.Length,table[3,2] ,table[3,6])
  p_Sepal.Width3 <- likelihood(x$Sepal.Width,table[3,3] , table[3,7])
  p_Petal.Length3 <- likelihood(x$Petal.Length,table[3,4] , table[3,8])
  p_Petal.Width3 <- likelihood(x$Petal.Width,table[3,5], table[3,9])
  return(p3=priorprob*p_Sepal.Length3*p_Sepal.Width3*p_Petal.Length3*p_Petal.Width3)
}

sample = data.frame(Sepal.Length=5.4, Sepal.Width=3.7, Petal.Length=1.5,Petal.Width=0.2) # A sample from Setosa (test data)
sample1 = data.frame(Sepal.Length=6.1, Sepal.Width=2.9, Petal.Length=4.7,Petal.Width=1.4) # A sample from Versicolor (test data)
sample2 = data.frame(Sepal.Length=6.3, Sepal.Width=2.9, Petal.Length=5.6,Petal.Width=1.8)# A sample from Virginica (test data)

#Below we call the functions for three classes of given sample set. For the highest probability, we check whether it is correctly 
#classified. 

#result <- data.frame(matrix(ncol = 4, nrow = 0))

result1=data.frame(posterior1(sample, p_setosa),
                  posterior2(sample, p_versicolor),
                  posterior3(sample, p_virginica))
result2=data.frame(Setosa=posterior1(sample1, p_setosa),
                  Versicolor=posterior2(sample1, p_versicolor),
                  Virginica=posterior3(sample1, p_virginica))

result3=data.frame(Setosa=posterior1(sample2, p_setosa),
                  Versicolor=posterior2(sample2, p_versicolor),
                  Virginica=posterior3(sample2, p_virginica))
result <- rbind(result1,result2,result3)
x<- c("Setosa","Versicolor","Virginica")
colnames(result) <- x

result
max<-c(max(result[1,]),max(result[2,]),max(result[3,]))
result$max<-as.data.frame(max)
result

#If you observe the max value in each row, it corresponds the same class we choosed our sample.  
# The aim of this learning material is to show the intuition behind Naive Bayes Method. The coding part is not optimized. Instead
#I tried to show every step.
