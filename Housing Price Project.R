## ---- include=FALSE, echo=FALSE----------------------------------------------------------------------------------------------------
library(ISLR)
library(plyr)
library(tidyr)
library(dplyr)
library(tree)
library(ggplot2)
library(reshape2)
library(class)
library(randomForest)


## ---- include=FALSE, echo=FALSE----------------------------------------------------------------------------------------------------
#Read in dataset 
temp1 <- read.csv(file = 'train.csv')
#head(temp1)

#Dimensions of original dataset
dim(temp1)


## ---- include = FALSE, echo=FALSE--------------------------------------------------------------------------------------------------
housing = subset(temp1, select = c(LotArea, OverallQual, YearRemodAdd, OverallCond, BedroomAbvGr
, YearBuilt, SalePrice) )

averageHPrice = sum(housing$SalePrice)/1460
housing$SaleClass <- as.factor(ifelse(housing$SalePrice <= averageHPrice , 'Cheap', 'Expensive'))

#Dimensions of dataset after we processed.
head(housing)


## ---- , echo=FALSE-----------------------------------------------------------------------------------------------------------------
#Exploratory Graphics

#Density Function
d <- density(housing$SalePrice)
plot(d, xlim = c(0,1000000), main="Distribution of Sale Prices")

#Boxplot Year Built, Sale Class
boxplot(YearBuilt~SaleClass, ylim = c(1850, 2010), data=housing, main= "Boxplot of Distribution of Year Built on Sale Class")

#Boxplot Overall Quality, Sale Class
boxplot(OverallQual~SaleClass,ylim = c(0, 10), data=housing, main="Boxplot of Distribution of Overall Quality on Sale Class")

#Boxplot LotArea, Sale Class
boxplot(LotArea~SaleClass, ylim = c(0, 50000), data=housing, main="Boxplot of Distribution of LotArea on Sale Class")

#Boxplot Remodeled, Sale Class
boxplot(YearRemodAdd~SaleClass, data=housing, main="Boxplot of Distribution of Year Remodeled on Sale Class")

#Boxplot Bedrooms, Sale Class
boxplot(BedroomAbvGr~SaleClass, data=housing, main="Boxplot of Distribution of Bedrooms on Sale Class")


## ---- include = FALSE, echo=FALSE--------------------------------------------------------------------------------------------------
# initializing a matrix for records
compare <- matrix(NA, nrow = 5, ncol = 3)
colnames(compare) <- c("Accuracy Rate", "Error Rate", "AUC")
rownames(compare) <- c("tree", "pruned.tree", 
                       "k=38 kNN", "k=15 kNN", 
                       "randomForest")


## ---- include = FALSE, echo=FALSE--------------------------------------------------------------------------------------------------
#Split into training and test set
set.seed(5)
group <- sample(1:2,nrow(housing),replace=TRUE,prob=c(.5,.5))
table(group)


## ---- include = FALSE, echo=FALSE--------------------------------------------------------------------------------------------------
housing.train <- housing[group==1,]
housing.test <- housing[group==2,]
head(housing.train)
head(housing.test)


## ---- include = FALSE, echo=FALSE--------------------------------------------------------------------------------------------------
# library(tree) is loaded
# predicting the label (good vs bad)
fit <- tree(SaleClass ~.-SalePrice, data = housing.train)


## ---- echo=FALSE-------------------------------------------------------------------------------------------------------------------

#$ variables actually used in tree construction"OverallQual"  "LotArea"      "YearBuilt"    "YearRemodAdd"
summary(fit)

plot(fit, type="uniform")
text(fit, pretty = 0, cex = 0.7, col = "red")
title("Classification Tree (Before Pruning)")


## ---- include = FALSE, echo=FALSE--------------------------------------------------------------------------------------------------
yhat.test <- predict(fit, housing.test, type="class")


## ---- echo=FALSE-------------------------------------------------------------------------------------------------------------------
# Obtain confusion matrix
error <- table(yhat.test, housing.test$SaleClass)
error

# Test accuracy rate
testacc = sum(diag(error))/sum(error)
testacc
# Test error rate (Classification Error)
testerr = 1-sum(diag(error))/sum(error)
testerr



## ---- warning=FALSE, echo=FALSE----------------------------------------------------------------------------------------------------
#ROC CURVE code here
library(ROCR)
# library(ROCR) is loaded
# getting matrix of predicted class probabilities
all_tree_probs <- as.data.frame(predict(fit, housing.test, type = "vector"))
tree_probs <- all_tree_probs[,2]

tree_roc_pred <- prediction(tree_probs, housing.test$SaleClass)
tree_roc_perf <- performance(tree_roc_pred, "tpr", "fpr")

# Plotting the ROC curve for the decision tree
plot(tree_roc_perf, col = 2, lwd = 3, 
     main = "ROC Curve for tree (before pruning)")
abline(0,1)

# Area under the curve
tree_auc_perf <- performance(tree_roc_pred, "auc")
tree_AUC <- tree_auc_perf@y.values[[1]]
tree_AUC

# adding to records matrix
compare[1, ] <- c(testacc, testerr, tree_AUC)


## ---- include = FALSE, echo=FALSE--------------------------------------------------------------------------------------------------
# library(tree) is loaded
cv <- cv.tree(fit, FUN=prune.misclass, K=6)
cv


## ---- echo=FALSE-------------------------------------------------------------------------------------------------------------------
# Best size 
best.cv <- cv$size[which.min(cv$dev)]

# plotting misclass error as a function of tree size (k)
plot(cv$size , cv$dev, type="b", 
     xlab = "Number of leaves, \'best\'", 
     ylab = "Misclassification Error",
     col = "red", main="Optimal Tree Size")
abline(v=best.cv, lty=2)


## ---- include = FALSE, echo=FALSE--------------------------------------------------------------------------------------------------
best.cv


## ---- echo=FALSE-------------------------------------------------------------------------------------------------------------------
tree.pruned <- prune.tree(fit, best = best.cv, 
                          method = "misclass")
ptree <- prune.misclass(fit,best=best.cv)
plot(ptree, type = "uniform")
text(ptree,pretty=0, cex = 0.7, col = "red")
title("Classification Tree (After Pruning)")



## ---- echo=FALSE-------------------------------------------------------------------------------------------------------------------
pruned_pred <- predict(tree.pruned, housing.test, type = "class")
# confusion matrix
pruned_conf <- table(pred = pruned_pred, true = housing.test$SaleClass)
pruned_conf
# Test accuracy rate
prunedacc = sum(diag(pruned_conf))/sum(pruned_conf)
prunedacc

# Test error rate (Classification Error)
prunederr = 1-sum(diag(pruned_conf))/sum(pruned_conf)
prunederr


## ---- echo=FALSE-------------------------------------------------------------------------------------------------------------------
all_pruned_probs <- as.data.frame(predict(tree.pruned, housing.test, type = "vector"))
pruned_probs <- all_pruned_probs[,2]

pruned_roc_pred <- prediction(pruned_probs, housing.test$SaleClass)
pruned_roc_perf <- performance(pruned_roc_pred, "tpr", "fpr")

# Plotting the ROC curve for the decision tree
plot(pruned_roc_perf, col = 2, lwd = 3, 
     main = "ROC Curve for tree (after pruning)")
abline(0,1)

# Area under the curve
pruned_auc_perf <- performance(pruned_roc_pred, "auc")
pruned_AUC <- pruned_auc_perf@y.values[[1]]
pruned_AUC

# adding to records matrix
compare[2, ] <- c(prunedacc, prunederr, pruned_AUC)


## ---- include = FALSE, echo=FALSE--------------------------------------------------------------------------------------------------
# Set random seed
#training and test set
xtrain <- housing.train[,c('LotArea', 'OverallQual', 'YearRemodAdd', 'OverallCond', 'BedroomAbvGr', 'YearBuilt')]
ytrain <- housing.train[,'SaleClass']
xtest <- housing.test[,c('LotArea', 'OverallQual', 'YearRemodAdd', 'OverallCond', 'BedroomAbvGr', 'YearBuilt')]
ytest <- housing.test[,'SaleClass']

#scale
xtrain <- scale(xtrain)
xtest <- scale(xtest,center=attr(xtrain,'scaled:center'),scale=attr(xtrain,'scaled:scale'))



## ---- include = FALSE, echo=FALSE--------------------------------------------------------------------------------------------------
#test k=38
pred.YTtrain38 <- knn(xtrain,xtrain,ytrain,k=38)
pred.YTest38 <- knn(xtrain,xtest,ytrain,k=38)

#conf matrix k=38
tab <- table(obs=ytrain,pred=pred.YTtrain38)
tab

#accuracy rate
sum(diag(tab)/sum(tab))

# misclassification error rate (training set) k=38
1 - sum(diag(tab)/sum(tab))


## ---- echo=FALSE-------------------------------------------------------------------------------------------------------------------
#test set conf matrix k=38
tab1 <- table(obs=ytest,pred=pred.YTest38)
tab1

#accuracy rate k=38
knnacc = sum(diag(tab1)/sum(tab1))
knnacc

# misclassification error rate (validation set) k=38
knnerr = 1 - sum(diag(tab1)/sum(tab1))
knnerr


## ---- include=FALSE, echo=FALSE----------------------------------------------------------------------------------------------------
#test k=39
pred.YTtrain39 <- knn(xtrain,xtrain,ytrain,k=39)
pred.YTest39 <- knn(xtrain,xtest,ytrain,k=39)

#conf matrix k=39
tab <- table(obs=ytrain,pred=pred.YTtrain39)
tab

#accuracy rate
sum(diag(tab)/sum(tab))

# misclassification error rate (training set) k=39
1 - sum(diag(tab)/sum(tab))


## ---- echo=FALSE-------------------------------------------------------------------------------------------------------------------
#test set conf matrix k=39
tab2 <- table(obs=ytest,pred=pred.YTest39)
tab2

#accuracy rate
sum(diag(tab2)/sum(tab2))

# misclassification error rate (validation set) k=39
1 - sum(diag(tab2)/sum(tab2))

compare[3, ] <- c(knnacc, knnerr, "NA")


## ---- include = FALSE, echo=FALSE--------------------------------------------------------------------------------------------------
#k-fold CV
do.chunk <- function(chunkid, folddef, Xdat, Ydat, ...){ # Function arguments
train = (folddef!=chunkid) # Get training index
Xtr = Xdat[train,] # Get training set by the above index
Ytr = Ydat[train] # Get true labels in training set
Xvl = Xdat[!train,] # Get validation set
Yvl = Ydat[!train] # Get true labels in validation set
predYtr = knn(train=Xtr, test=Xtr, cl=Ytr, ...) # Predict training labels
predYvl = knn(train=Xtr, test=Xvl, cl=Ytr, ...) # Predict validation labels
data.frame(fold = chunkid, # k folds
train.error = mean(predYtr != Ytr), # Training error for each fold
val.error = mean(predYvl != Yvl)) # Validation error for each fold
}


## ---- include = FALSE, echo=FALSE--------------------------------------------------------------------------------------------------
nfolds <- 5
folds = sample(cut(1:nrow(xtrain), breaks=nfolds, labels=FALSE))
table(folds)

head(folds,5)


## ---- include = FALSE, echo=FALSE--------------------------------------------------------------------------------------------------
error.folds = NULL
kvec <- 1:50
for( j in kvec ) {
 tmp = ldply(1:nfolds, do.chunk,folddef=folds, Xdat=xtrain, Ydat=ytrain, k=j)
  tmp$neighbors = j 
  error.folds = rbind(error.folds,tmp) 
}
dim(error.folds)


## ---- include = FALSE, echo=FALSE--------------------------------------------------------------------------------------------------
head(error.folds)


## ---- include = FALSE, echo=FALSE--------------------------------------------------------------------------------------------------
# Transform the format of error.folds for further convenience
errors = melt(error.folds, id.vars=c('fold', 'neighbors'), value.name='error')
# Choose the number of neighbors which minimizes validation error
val.error.means = errors %>%
# Select all rows of validation errors
filter(variable=='val.error') %>%
# Group the selected data frame by neighbors
group_by(neighbors, variable) %>%
# Calculate CV error rate for each k
summarise_each(funs(mean), error) %>%
# Remove existing group
ungroup() %>%
filter(error==min(error))


## ----------------------------------------------------------------------------------------------------------------------------------
# Best number of neighbors
# if there is a tie, pick larger number of neighbors for simpler model
numneighbor = max(val.error.means$neighbors)
numneighbor


## ---- echo=FALSE-------------------------------------------------------------------------------------------------------------------
pred.YTest = knn(train=xtrain, test=xtest, cl=ytrain, k=numneighbor)
# Confusion matrix
conf.matrix = table(predicted=pred.YTest, true=ytest)
conf.matrix

# Test accuracy rate
bknnacc = sum(diag(conf.matrix)/sum(conf.matrix))
bknnacc
# Test error rate
bknnerr = 1 - sum(diag(conf.matrix)/sum(conf.matrix))
bknnerr

compare[4, ] <- c(bknnacc, bknnerr, "NA")


## ---- include=FALSE, echo=FALSE----------------------------------------------------------------------------------------------------
### Random Forest
# using all 6 predictor attributes, on the training set
rf <- randomForest(formula = SaleClass ~ .-SalePrice,
                   data = housing.train,
                   mtry = 6)

print(rf)

## ---- echo=FALSE-------------------------------------------------------------------------------------------------------------------
varImpPlot(rf, main = "Variable Importance Plot")


## ---- echo=FALSE-------------------------------------------------------------------------------------------------------------------
# predicting on the test set
rf_pred <- predict(rf, housing.test, type = "class")

# Confusion Matrix
rf_conf <- table(true = housing.test$SaleClass, pred =  rf_pred)
rf_conf


## ---- echo=FALSE-------------------------------------------------------------------------------------------------------------------
#accuracy rate
rf_acc = sum(diag(rf_conf))/sum(rf_conf)
rf_acc

#error rate
rf_err <- 1 - rf_acc
rf_err


## ---- echo=FALSE-------------------------------------------------------------------------------------------------------------------
# Building the ROC Curve
rf_pred <- as.data.frame(predict(rf, newdata = housing.test, type = 'prob'))
rf_pred_probs <- rf_pred[,2]
rf_roc_pred <- prediction(rf_pred_probs, housing.test$SaleClass)
rf_perf <- performance(rf_roc_pred, measure = "tpr", 
                       x.measure = "fpr")

# Plotting the curve
plot(rf_perf, col = 2, lwd = 3, 
     main = "ROC Curve for randomForest with all 6 variables")
abline(0,1)

# Area under the curve
rf_perf2 <- performance(rf_roc_pred, measure = "auc")
rf_AUC <- rf_perf2@y.values[[1]]
rf_AUC

compare[5, ] <- c(rf_acc, rf_err, rf_AUC)


## ----------------------------------------------------------------------------------------------------------------------------------
compare

