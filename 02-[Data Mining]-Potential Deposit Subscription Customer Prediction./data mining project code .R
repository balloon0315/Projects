#input data 

setwd('C:/Users/Weidan He/Desktop/DM project/Project_submit')
mydata=read.table('bank-full.csv', header=T, sep=';')
summary(mydata)

#check variable
str(mydata)
table(mydata$y)
sum(is.na(mydata))

#feature selection
#for the categorical variables, we use Chi-square test to check the independence.
library(MASS)

#education
tb_education = table(mydata$education,mydata$y) 
tb_education               
chisq.test(tb_education)

# test the unknown values of education
known=tb_education['primary',] + tb_education['secondary',]+tb_education['tertiary',]
unknown=tb_education['unknown',]
tb1=rbind(known,unknown)
tb1
chisq.test(tb1)

#job
tb_job = table(mydata$job,mydata$y) 
tb_job               
chisq.test(tb_job)

#marital
tb_marital = table(mydata$marital,mydata$y) 
tb_marital               
chisq.test(tb_marital)

#default
tb_default = table(mydata$default,mydata$y) 
tb_default               
chisq.test(tb_default)

#housing
tb_housing = table(mydata$housing,mydata$y) 
tb_housing              
chisq.test(tb_housing)

#loan
tb_loan = table(mydata$loan,mydata$y) 
tb_loan               
chisq.test(tb_loan)

#contact
tb_contact = table(mydata$contact,mydata$y) 
tb_contact               
chisq.test(tb_contact)

# test the unknown values of contact
known=tb_contact['cellular',] + tb_education['telephone',]
unknown=tb_contact['unknown',]
tb2=rbind(known,unknown)
tb2
chisq.test(tb2)

#month
tb_month = table(mydata$month,mydata$y) 
tb_month               
chisq.test(tb_month)

#poutcome
tb_poutcome = table(mydata$poutcome,mydata$y) 
tb_poutcome               
chisq.test(tb_poutcome)

# test the unknown values of poutcome
known=tb_poutcome['failure',] + tb_poutcome['other',]+tb_poutcome['success',]
unknown=tb_poutcome['unknown',]
tb3=rbind(known,unknown)
tb3
chisq.test(tb3)


# for the numeric variables, we use Pearson correlation method.
myvars=c("age", "balance", "day","duration","campaign","pdays","previous","y")
datacorr=mydata[myvars]
datacorr$y=as.numeric(datacorr$y)
tail(cor(datacorr),1)                 #correlation
tail(round(cor_pmat(datacorr),3),1)   #P-value


#stratified sampling 
install.packages("caret")
install.packages("lattice")
install.packages('ggplot2')
install.packages('Rcpp')
library(caret)
library(lattice)
library(ggplot2)
library(Rcpp)
ince=createDataPartition(y=mydata$y,p=0.7,list=FALSE)
trainset=mydata[ince,]
testset_all=mydata[-ince,]

#Extract the testlabel and create the testset
testLabel=testset_all$y
testset=mydata[-ince,-17]

#Logistic regression
fit1=glm(trainset$y~., data=trainset, family = binomial)
select=step(fit1,direction="backward")
fit.final=glm(trainset$y~job + marital + education + balance + housing + loan + contact + day + month + duration + campaign + poutcome, data = trainset,family = binomial)
Pre_label1=ifelse(predict(fit.final,testset, type = "response") > 0.5, "yes", "no")
Confusion_Table=table(predicted = Pre_label1, actual = testLabel)
##the accuracy of Logistic regression
confusionMatrix(Confusion_Table)
## ROC and AUC
install.packages('pROC')
library(pROC)
test_prob1=predict(fit.final,testset, type = "response")
Log.Roc=roc(testLabel ~ test_prob1, plot = TRUE, print.auc = TRUE)

#svm
library("e1071")
fit2=svm(trainset$y~.,data=trainset)
Pre_label2=predict(fit2,testset)
##the accuracy of SVM
confusionMatrix(Pre_label2, testLabel)
## ROC and AUC 
library("pROC")
svm.ROC = roc(testLabel ~ as.numeric(Pre_label2), plot=TRUE,print.auc = TRUE)

#Naive bayes
library("e1071")
fit3=naiveBayes(trainset$y~.,data=trainset)
Pre_label3=predict(fit3,testset)
##the accuracy of Naive bayes
confusionMatrix(Pre_label3, testLabel)
## ROC and AUC
library("pROC")
NB.ROC = roc(testLabel ~ as.numeric(Pre_label3), plot=TRUE,print.auc = TRUE)

#Based on the comparison of the three models, we can tell that Logistic regression model is 
#the best. Now we can get the model by logistic regression model.

#Get the coefficiencts of the best model
summary(fit.final)
sort(coef(fit.final),decreasing = T)

# Using varImp() function to get the importance of the variables
library(caret)
Importance=varImp(fit.final)


 