knitr::opts_chunk$set(echo = FALSE, include = FALSE)

##################
### Question 1 ###
##################

## Clear environment
rm(list = ls()) # removes all variables
if(!is.null(dev.list())) dev.off() # clear plots
cat("\014") # clear console

## Import required packages
library(caret, warn.conflicts = FALSE, quietly = TRUE) # handy ml package, data splitting, training ect ect
library(tidyverse, warn.conflicts = FALSE, quietly = TRUE) # handy for data prep
library(reshape2, warn.conflicts = FALSE, quietly = TRUE) # handy for melt() - plotting data frames with ggplot
library(alookr, warn.conflicts = FALSE, quietly = TRUE) # for removing correlated variables
library(ggplot2, warn.conflicts = FALSE, quietly = TRUE) # plotting
library(ggpubr, warn.conflicts = FALSE, quietly = TRUE) # added plotting function
library(DataExplorer, warn.conflicts = FALSE, quietly = TRUE) # quick exploratory vis
library(corrplot, warn.conflicts = FALSE, quietly = TRUE) # plotting corrmatrix
library(ROCR, warn.conflicts = FALSE, quietly = TRUE) # for ROC curves
library(ROCit, warn.conflicts = FALSE, quietly = TRUE) # for ROC curves


file <- 'Breast_cancer.csv' # store the path to the source data
rawData <- read.csv(file, header = TRUE, stringsAsFactors = TRUE) # import source data, in this case the data file has headers and strings will be used a factors
names(rawData) <- c("id", "diagnosis", "radius", "texture","perimeter", "area", "smoothness", "compactness", "concavity", "concave.points", "symmetry",  "fractal_dimension") # set column names to meaningful titles
rawData <- within(rawData, rm("id")) # remove variable ID as it is simply a record identification


## initial exploratory vis, stored as variables to display in appendix 1
a1_1_intro <- introduce(rawData) # provides introductory summary, good to confirm variable types, missing data ect
a1_2_head <- head(rawData) # handy to have a look at actual data
a1_3_dataForAppendix <- rawData

## standardize data so as it has mean 0 and variance 1
rD_stz <- preProcess(rawData[ ,-1], method = c("scale", "center")) # set parameters for scaling data
rawData_stz <- predict(rD_stz, rawData) # scale data
a1_4_sum_after_stz <- summary(rawData_stz) # save summary to display in appendix


## plot correlation matrix to inspect for correlated variables
corMatrix <- round(cor(rawData_stz[ ,-1], method = "pearson"), 2) # calculate correlation matrix, using pearson
corrplot.mixed(corMatrix, order = "AOE") # plot correlation matrix


## rough and ready glm and calculation of log odds for plots and predictor assessment
logistic_initial <- glm(diagnosis ~ . , data = rawData_stz, family = binomial(link = "logit")) # fit a glm model, binomial as we have only 2 responses classes, using the logit link function
probs_initial <- predict(logistic_initial, type = "response") # use model to calculate the probabilities of the positive outcome 
predictors <- names(rawData_stz[ ,-1]) # names of predictor variables
odds <- rawData_stz %>% mutate(logit = log(probs_initial / (1 - probs_initial))) # create new column with the log odds for each case
odds <- melt(odds[ ,-1], id.vars = "logit") # melt data for plotting

## Plot predictors vs log odds to check linearity assumption
ggplot(odds, aes(logit, value)) + geom_point(size = 0.5, alpha = 0.5) + facet_wrap(~variable) + geom_smooth(formula = y ~ x, method = "loess") + ggtitle("Predictor Variables vs Log Odds of Response Variable") # create a grid of plots with log odds on x axis and the value on the y

## frequency distributions to understand distribution of predictors by response class
ggplot(data = melt(rawData_stz, id.var = "diagnosis" ), mapping = aes(x = value, fill = diagnosis)) + geom_density(alpha = 0.5) + facet_wrap(~variable, scales = "free") + ggtitle("Density distribution by diagnosis for all predictors")


## Replot correlation matrix to ensure suitab;e, save to display in appendix 1
rawData_drop <- within(rawData_stz, rm("perimeter", "area", "compactness", "concave.points")) # drop correlated variables
a1_5_cor <- round(cor(rawData_drop[ ,-1]), 2) # create correlation matrix

## rough and ready glm for predictor assessment
logistic_influence <- glm(diagnosis ~ . , data = rawData_drop, family = binomial(link = "logit")) # fit a glm model, binomial as we have only 2 responses classes, using the logit link function

## plot cooks's distance
plot(logistic_influence, which = 4, id.n = 5) # plot cooks distance

# house keeping
modelData <- rawData_drop # move data into modelData data frame
rm(corMatrix, odds, rD_stz, file, predictors, probs_initial, rawData, rawData_drop, rawData_stz, logistic_influence, logistic_initial) # remove variables that are not required.


### Prepare and fit final model
## Create test / training split
train_index <- createDataPartition(modelData$diagnosis, p=0.8, list = FALSE) # returns numerical vector of the index of the observations to be included in the training set
testData <- modelData[-train_index, ] # create data.frame of test data
trainingData <- modelData[train_index, ] # create data frame of training predictors

## fit logistic regression model
bc_logistic <- glm(diagnosis ~ . , data = trainingData, family = binomial(link = "logit")) # fit logistic regresion model

## return some information for assessing the model
prob_train <- predict(bc_logistic, type = "response") # calculate probabilities of each case in the training data being "M"
prob_train <- ifelse(prob_train > 0.5, "M", "B") # convert probability into classification assuming .5 descion boundary
confMatrix_train <- confusionMatrix(as.factor(prob_train), trainingData$diagnosis) # create confusion matrix summarizing accuracy on training data


## crsave assesment values for displaying in appendix
ROCpred <- prediction(predict(bc_logistic, type = "response"), trainingData$diagnosis)
a1_6_roc <- performance(ROCpred, "tpr", "fpr") # save values for roc curve
a1_7_1_dev <- bc_logistic$deviance # save deviance
a1_7_2_nullDev <- bc_logistic$null.deviance # save null deviance
a1_7_3_aic <- bc_logistic$aic # save aic 
a1_8 <- confMatrix_train # save confusion matrix


## Create plot of positive predictive power
PPV <- measureit(score = bc_logistic$fitted.values, class = bc_logistic$y, measure = c("PPV")) # creates object summarizing the PPV of the model for different cutoffs
PPV_plotDF <- data.frame(PPV$Cutoff, PPV$PPV) # convert to df for ggplot
names(PPV_plotDF) <- c("Probability Threshold", "Positive Predictive Power") # name data frame columns
ppv_plot <- ggplot(na.omit(PPV_plotDF), aes(`Probability Threshold`, `Positive Predictive Power`)) + geom_line(size = 1, colour = "red") + geom_hline(aes(yintercept=0.995), colour = "blue")+ geom_text(aes( 0, 0.995, label = 0.995, vjust = -1), size = 3, colour = "blue") + geom_vline(aes(xintercept=0.8), colour = "blue")+ geom_text(aes( .8, 0, label = 0.8, hjust = -1), size = 3, colour = "blue") + theme_light() + ggtitle("Positive Predictive power Vs Probability Threshold") # define plot
ppv_plot

prob_test <- predict(bc_logistic, newdata = testData, type = "response") # calculate probabilities of each case in the test data being "M"
prob_test_5 <- ifelse(prob_test > 0.5, "M", "B") # convert probability into classification assuming .5 descion boundary
prob_test_8 <- ifelse(prob_test > 0.8, "M", "B") # convert probability into classification assuming .5 descion boundary
confMatrix_test_5 <- confusionMatrix(as.factor(prob_test_5), testData$diagnosis) # create confusion matrix summarizing accuracy on training data
confMatrix_test_8 <- confusionMatrix(as.factor(prob_test_8), testData$diagnosis) # create confusion matrix summarizing accuracy on training data
a1_9_cor5 <- confMatrix_test_5$table # save confusion matrix table for appendix
a1_10_cor8 <- confMatrix_test_8$table # save confusion matrix for appendix
confMatrix_test_5$byClass # return accuracy by class
confMatrix_test_8$byClass # return accuracy by class

## output p values 
summary(bc_logistic) # p values


##################
### Question 3 ###
##################

## Clear environment
rm(bc_logistic, confMatrix_test_5, confMatrix_test_8, confMatrix_train, modelData, PPV, ppv_plot, PPV_plotDF, prob_test, prob_test_5, prob_test_8, prob_train, ROCpred, testData, train_index, trainingData) # remove variables from last question, except appendix
if(!is.null(dev.list())) dev.off() # clear plots
cat("\014") # clear console

## Import required packages
library(caret, warn.conflicts = FALSE, quietly = TRUE) # handy ml package, data splitting, training ect ect
library(cluster, warn.conflicts = FALSE, quietly = TRUE) # for clustering
library(tidyverse, warn.conflicts = FALSE, quietly = TRUE) # handy for data prep
library(ggplot2, warn.conflicts = FALSE, quietly = TRUE) # plotting
library(ggpubr, warn.conflicts = FALSE, quietly = TRUE) # added plotting function
library(ggtext, warn.conflicts = FALSE, quietly = TRUE) # more plotting
library(DataExplorer, warn.conflicts = FALSE, quietly = TRUE) # quick exploratory vis
library(corrplot, warn.conflicts = FALSE, quietly = TRUE) # plotting corrmatrix
library(alookr, warn.conflicts = F, quietly = T) # for removing correlated variables
library(proxy, warn.conflicts = FALSE, quietly = TRUE) # for computing dissimilarity
library(factoextra, warn.conflicts = FALSE, quietly = TRUE) # visualizing clustering
library(ggdendro, warn.conflicts = FALSE, quietly = TRUE) # for some clever dendrograms

file <- 'leukemia_dat.csv' # store the path to the source data
rawData <- read.csv(file, header = TRUE, stringsAsFactors = TRUE) # import source data, in this case the data file has no headers and strings will be used a factors 
rawData <- within(rawData, rm("X", "patient_id")) # remove variable ID as it is simply a record identification


## initial exploratory vis, stored as variables to display in appendix 1
a2_1_intro <- introduce(rawData) # provides introductory summary, good to confirm variable types, missing data ect


rawData_corr <- treatment_corr(rawData, corr_thres = 0.9)


## Display variables that were removed and house keeping
(names(rawData)[! names(rawData) %in% names(rawData_corr)]) # print columns that where dropped in above steps
print("total # removed")
length((names(rawData)[! names(rawData) %in% names(rawData_corr)]))
actualClasses <- rawData_corr$type # save the actual classes for reference later
rawData_corr <- within(rawData_corr, rm("type")) # remove class variable so as it is not included in model training
modelData_df <- rawData_corr # move data into model data frame
modelData_matrix <- as.matrix(modelData_df) # create matrix of model data


## calculate dissimilarity matrix using various methods.
dissimilarityArray_all <- array(dim = c(72, 72, 3))
dissimilarityArray_all[ , ,1] <- as.matrix(dist(modelData_matrix, method = "cosine")) # dissimilarity based on cosine distance and save as the first matrix in array
dissimilarityArray_all[ , ,2] <- as.matrix(dist(modelData_matrix, method = "euclidean")) # dissimilarity based on euclidean distance and save as second matrix in array
dissimilarityArray_all[ , ,3] <- as.matrix(dist(modelData_matrix, method = "manhattan")) # dissimilarity based on manhattan distance and save as third matrix in array


## generate cluster assignment for all 3 dissimilarity matrix using single, complete, average and ward methods
names_dis <- c("cosine", "eucledian", "manhattan") # vector of matrix identifiers
methods <- c("single", "complete", "average", "ward") # vector of method types
count <- 0 # set counter
ac_value <- c() # empty vector to store Ac value for quick access later
ac_label <- c() # empty vector to stor AC labels
ac_sim <- c() # empty vector to store the matrix type for each ac score

for (i in 1:3) { # loop over array full of dissimilarity matrix 
  for (m in 1:4){ # loop over methods
    count <- count + 1 # increase counter by 1
    
    variable_id <- paste("cluster",names_dis[i], methods[m], sep = "_") # generate variable id
    model <- agnes(dissimilarityArray_all[ , ,i], diss = TRUE, method = methods[m]) # generate model
    assign(variable_id, model) # assign model to variable id
    ac_value[count] <- model$ac # record ac value
    ac_label[count] <- paste(names_dis[i], methods[m], sep = " ") # record label to label ac value
    ac_sim[count] <- methods[m] # record similarity matrix used
    
  }
}

acValue_all <- as.data.frame(cbind(ac_label, ac_value, ac_sim)) # make ac vectors into a data frame for plotting later


## use ggplot to make a column graph showing the model performance
agCoef_graph <- ggplot(acValue_all, aes(ac_label, ac_value, fill = ac_sim)) + geom_col(width = ac_value) + 
  theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust = 1), legend.position = "none") + 
  labs(x = "Model", y = "AC Value", title = "Aglomerative coefficent for all models")

agCoef_graph # display graph


# Cut tree at k = 2 to find the 2 classifications found by the model
modelResults <- cutree(cluster_cosine_ward, k = 2) # cut tree at k  =2
modelResults <- as.factor(ifelse(modelResults == 1, "ALL", "AML")) # if cluster is 1 then assign ALL otherwise AML, change to factor for confusion matrix
confMatrix <- confusionMatrix(modelResults, actualClasses)
confMatrix


# Cut tree at k = i to find the swc for each option
swc <- data.frame() # create empty data frame hold swc values

for(i in 2:71){ # loop through all posible number of clusters
  sil <- silhouette(cutree(cluster_cosine_ward, k = i), dissimilarityArray_all[ , , 1]) # calculate SWC
  sil <- summary(sil) # extra summary so avg can be easily accessed
  swc[i-1,1] <- i # record number of clusters
  swc[i-1,2] <- sil$avg.width # record avg
}

## calculate gap stats
GAPFUN <- function(dissMatrix, k){ # define a function that takes a disimilarity matrix and number of clusters
  
  list(cluster = cutree(agnes(dissMatrix, diss = TRUE, method = "ward"), k = k)) # create a list of the cluster for each observation
}
gapStats <- clusGap(dissimilarityArray_all[ , ,1],GAPFUN, B = 50, K.max = 71 ) # calculate gap stats

evalPlot_df <- cbind(swc[1:25, ], gapStats$Tab[2:26,3])
names(evalPlot_df) <- c("# Clusters", "AVG Silhoutte Width", "Gap Statistic") # give Df columns names

## plot both with ggplot
eval_plot <- ggplot(evalPlot_df ) + 
  geom_point(aes(`# Clusters`, `AVG Silhoutte Width`), colour = "blue", size = 2, alpha = 0.75) + geom_line(aes(`# Clusters`, `AVG Silhoutte Width`), colour = "grey") +
  geom_point(aes(`# Clusters`, `Gap Statistic`), colour = "orange", size = 2, alpha = 0.75) + geom_line(aes(`# Clusters`, `Gap Statistic`), colour = "grey")+ 
  geom_vline(aes(xintercept=3), colour = "red")+ geom_text(aes( 3, 0, label = 3, hjust = -1), size = 3, colour = "red") + # vertical red line at point of interest
  geom_text(aes( 22, 0.9, label = "GAP", hjust = -1), size = 6, colour = "orange") +
  geom_text(aes( 22, 0.3, label = "SWC", hjust = -1), size = 6, colour = "blue") +
  scale_y_continuous("AVG Silhoutte Width", sec.axis = sec_axis(~ . * 1, name = "Gap Statistic")) +
  labs(title = " Silhoutte Width Criterion and Gap Statistic Vs # Cluster") +
  theme_minimal()
eval_plot # display plot


dendro <- as.dendrogram(cluster_cosine_ward)
actualClasses_df <- data.frame(c(1:72),actualClasses)
names(actualClasses_df) <- c("label", "Actual Class")
actualClasses_df$label <- as.character(actualClasses_df$label)
dendroDat <- dendro_data(dendro)
dendroSeg <- dendroDat$segments
dendroEnds <- dendroSeg %>% filter(yend == 0) %>% left_join(dendroDat$labels, by = "x") %>% left_join(actualClasses_df, by = "label")
classColour <- c("ALL" = "blue", "AML" = "red")
                 
dendro1 <- ggplot() + geom_segment(data = dendroSeg, aes(x=x, y=y, xend=xend, yend=yend), size = 01, alpha = 0.75, colour = "dark grey") + 
  geom_segment(data = dendroEnds, aes(x=x, y=y.x, xend=xend, yend=yend, color = `Actual Class`), size = .8, alpha = 0.75) + 
  geom_richtext(data = dendroEnds, aes(x = x, y = 0, label = label), size = 3, angle = 90) +
  scale_color_manual(values = classColour) + 
  ylab("Distance") + 
  xlab("Patient ID") +
  ggtitle("Leukemia Classification Dendrogram") + 
  geom_rect(aes(xmin = 0, xmax = 49.5, ymin = 0, ymax = 2.5), fill = "blue", alpha = 0.1) + 
  geom_rect(aes(xmin = 49.55, xmax = 73, ymin = 0, ymax = 1.5), fill = "red", alpha = 0.1) +
  geom_text(aes(x = 5, y = 2.25, label = "Classified as ALL"), size = 4, colour = "blue") +
  geom_text(aes(x = 67, y = 1.25, label = "Classified as AML"), size = 4, colour = "red") +
  theme_minimal()+
  theme(axis.text.x = element_blank())
  
dendro1


a1_1_intro


a1_2_head


str(a1_3_dataForAppendix)


a1_4_sum_after_stz


corrplot.mixed(a1_5_cor, order = "AOE")


plot(a1_6_roc, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.2), text.adj=c(-0.2,1.7)) + title(main = "ROC Curve")


a1_7_1_dev

a1_8


a1_9_cor5


a1_10_cor8


## initial exploratory vis, stored as variables to display in appendix 1
a2_1_intro

