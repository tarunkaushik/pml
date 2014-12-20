# Activity quality prediction - ML algorithm

**Overview** : The aim of the analysis is to predict the quality of activity performed. For this data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants was collected. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The data (source: http://groupware.les.inf.puc-rio.br/har) contains various measurements from which we have to build the machine learning algorithm.

## Question
Given a set of features we have to determine the manner in which activity was performed.

## Input data
The data was obtained from http://groupware.les.inf.puc-rio.br/har in two sets.
Data for training contained 19622 observations.
We have to build and test our algorithm on this dataset. column classe in the dataset is the one we have to predict using rest of the 159 columns.
One more dataset, testing was also provided, for which we have to predict the classe. There are only 20 observations in the dataset.

Data and libraries were loaded into R


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(xtable)
```

```
## Warning: package 'xtable' was built under R version 3.1.2
```

```r
library(ggplot2)
trainData <- read.csv("pml-training.csv")
validation <- read.csv("pml-testing.csv")
```

## Features
For each observation, there are 160 variables, out of which one is classe, which we have to predict. We can use rest of the 159 features. 
We have to identify the features which we might eliminate while fitting the model.
there are two kinds of features which we want to eliminate.
* Features which have values missing for all the observations.
* Features which do not have any relation with classe.

Our aim in the analysis is to determine classe for the 20 observations in dataset testing. Hence features which have missing values for all the observations were identified and removed from each dataset.
After removing such features it was observed that no other feature in either of the dataset had missing value for any of the observations. Voila! No imputation needed.

Next varaibles such as serial number, date and time stamps were removed as there is no reason why they should have any predictive power.

Following functions were used in subsequent code.

```r
booleanToInt <- function(value)
{
	value*1
}

zeroOrNot <- function(value)
{
	if(value==0)
	{
		0
	} else {
		1
	}
}
```

Final set of features was selected using the code below.

```r
featureNames <- names(trainData)

k <- data.frame(sapply(validation, FUN=is.na))
k <- data.frame(sapply(k,FUN=booleanToInt))
k <- data.frame(sapply(k,FUN=sum))
names(k) <- "isNA"
cn <- rep(NA,dim(validation)[2])
for(i in 1:dim(validation)[2])
{
	cn[i] <- if(k$isNA[i]!=0) { 1 } else { 0 }
}

cn2 <- rep(NA,dim(validation)[2])
for(i in 1:dim(validation)[2])
{
	cn2[i] <- i*(1-cn[i])
}

cn2 <- cn2[cn2>0]

cn2 <- setdiff(cn2,c(1,3,4,5))

trainData <- trainData[cn2]
validation <- validation[cn2]
```

We were left with a set of 55 features which we could use to build the algorithm.
Since classification models were to be fit, PCA was not used to pre-process the features. Also features were not normalized for the same reason.


## Algorithm
Since the variable we need to predict is not numeric, we need classification techniques.
Random forest has been used here.
For this first trainData was split in two different segments, namely
* training
* testing

The random forest model needs to be fit on training data set and validated on testing dataset.
Since we have to run the model multiple times, so as to pick the best performing model, we can't use whole training data to fit the model and test the performance on testing dataset.
Hence for cross validation I have used the following strategy.

Cross-validation technique used: **Random subsampling**

Training dataset was split in training and testing 5 times, i.e.
* training1 and testing1
* training2 and testing2
* training3 and testing3
* training4 and testing4
* training5 and testing5


```r
set.seed(2212)
inTrain = createDataPartition(trainData$classe, p = 0.6)[[1]]
training = trainData[inTrain,]
testing = trainData[-inTrain,]

set.seed(45115)
seeds <- round((10000*rnorm(5,mean=100,sd=10)^2)^0.5)

set.seed(seeds[1])
inTrain1 = createDataPartition(training$classe, p = 0.6)[[1]]
set.seed(seeds[2])
inTrain2 = createDataPartition(training$classe, p = 0.6)[[1]]
set.seed(seeds[3])
inTrain3 = createDataPartition(training$classe, p = 0.6)[[1]]
set.seed(seeds[4])
inTrain4 = createDataPartition(training$classe, p = 0.6)[[1]]
set.seed(seeds[5])
inTrain5 = createDataPartition(training$classe, p = 0.6)[[1]]

training1 = training[inTrain1,]
testing1 = training[-inTrain1,]
training2 = training[inTrain2,]
testing2 = training[-inTrain2,]
training3 = training[inTrain3,]
testing3 = training[-inTrain3,]
training4 = training[inTrain4,]
testing4 = training[-inTrain4,]
training5 = training[inTrain5,]
testing5 = training[-inTrain5,]
```

### Feature plots
We are building a machine learning algorithm, hence interpretability is not as much important here. Still we would look at plots of some of the features. (on training data, keeping testing data aside)



```r
qplot(training[,3],colour=classe, data=training, geom="density", xlab=names(training)[3])
```

![plot of chunk FeaturePlot1](figure/FeaturePlot1-1.png) 


```r
qplot(training[,17],colour=classe, data=training, geom="density", xlab=names(training)[17])
```

![plot of chunk FeaturePlot2](figure/FeaturePlot2-1.png) 


```r
qplot(training[,20],colour=classe, data=training, geom="density", xlab=names(training)[20])
```

![plot of chunk FeaturePlot3](figure/FeaturePlot3-1.png) 


```r
qplot(training[,24],colour=classe, data=training, geom="density", xlab=names(training)[24])
```

![plot of chunk FeaturePlot4](figure/FeaturePlot4-1.png) 

We can see that there is slight difference among distributon of feature values across different classe. We expect our machine learning technique to identify all such subtle differences and predict the outcome accurately.


## Cross validation - Picking up best performing model

Each time a random forest model was fit on training dataset and accuracy was determined on paired testing dataset.
The model with best accuracy would be used as the FinalModel.
We would validate the final model on our initial testing dataset to get an idea of the out of sample error-rate. we would do this only once, at the end.
In the end we would use this final model to predict classe of our validation dataset.


```r
modelFit1 <- train(classe~.,data=training1, method="rf")
```

```
## Error in eval(expr, envir, enclos): could not find function "train"
```

```r
modelFit2 <- train(classe~.,data=training2, method="rf")
```

```
## Error in eval(expr, envir, enclos): could not find function "train"
```

```r
modelFit3 <- train(classe~.,data=training3, method="rf")
```

```
## Error in eval(expr, envir, enclos): could not find function "train"
```

```r
modelFit4 <- train(classe~.,data=training4, method="rf")
```

```
## Error in eval(expr, envir, enclos): could not find function "train"
```

```r
modelFit5 <- train(classe~.,data=training5, method="rf")
```

```
## Error in eval(expr, envir, enclos): could not find function "train"
```
After fitting, each model was validated on corresponding testing dataset.
Overall summary of Confusion Matrix of each model helped in determining the best performing model out of the 5 models.


```r
testing1$pred=predict(modelFit1, testing1)
```

```
## Error in predict(modelFit1, testing1): object 'modelFit1' not found
```

```r
testing2$pred=predict(modelFit2, testing2)
```

```
## Error in predict(modelFit2, testing2): object 'modelFit2' not found
```

```r
testing3$pred=predict(modelFit3, testing3)
```

```
## Error in predict(modelFit3, testing3): object 'modelFit3' not found
```

```r
testing4$pred=predict(modelFit4, testing4)
```

```
## Error in predict(modelFit4, testing4): object 'modelFit4' not found
```

```r
testing5$pred=predict(modelFit5, testing5)
```

```
## Error in predict(modelFit5, testing5): object 'modelFit5' not found
```

```r
testing1Results <- confusionMatrix(testing1$pred,testing1$classe)$overall
```

```
## Error in confusionMatrix.default(testing1$pred, testing1$classe): the data and reference factors must have the same number of levels
```

```r
testing2Results <- confusionMatrix(testing2$pred,testing2$classe)$overall
```

```
## Error in confusionMatrix.default(testing2$pred, testing2$classe): the data and reference factors must have the same number of levels
```

```r
testing3Results <- confusionMatrix(testing3$pred,testing3$classe)$overall
```

```
## Error in confusionMatrix.default(testing3$pred, testing3$classe): the data and reference factors must have the same number of levels
```

```r
testing4Results <- confusionMatrix(testing4$pred,testing4$classe)$overall
```

```
## Error in confusionMatrix.default(testing4$pred, testing4$classe): the data and reference factors must have the same number of levels
```

```r
testing5Results <- confusionMatrix(testing5$pred,testing5$classe)$overall
```

```
## Error in confusionMatrix.default(testing5$pred, testing5$classe): the data and reference factors must have the same number of levels
```

```r
testingResults <- rbind(testing1Results,testing2Results,testing3Results,testing4Results,testing5Results)
```

```
## Error in rbind(testing1Results, testing2Results, testing3Results, testing4Results, : object 'testing1Results' not found
```

```r
testingResults <- data.frame(testingResults)
```

```
## Error in data.frame(testingResults): object 'testingResults' not found
```

```r
modelFit1Accuracy <- round(testingResults$Accuracy[1]*100,2)
```

```
## Error in eval(expr, envir, enclos): object 'testingResults' not found
```

```r
testingResults <- xtable(data.frame(testingResults))
```

```
## Error in data.frame(testingResults): object 'testingResults' not found
```

```r
print(testingResults,type="html")
```

```
## Error in print(testingResults, type = "html"): object 'testingResults' not found
```







