a <- read.table("C:/Users/max22/Desktop/N_H_e_Validation1.txt")
a <- as.matrix(a)
label <- a[,2]
decision <- a[,1]
ngrids <- 100
TPR <- rep(0, ngrids)
FPR <- rep(0, ngrids)
p0 <- rep(0, ngrids)

for(i in 1:ngrids)
{
  p0[i] <- i/ngrids
  pred_label <- 1*(decision > p0[i])
  TPR[i] <- sum(pred_label * label) / sum(label)
  FPR[i] <- sum(pred_label * (1-label)) / sum(1-label)
}


## calculate AUC

pos.decision <- decision[which(label == 1)]
neg.decision <- decision[which(label == 0)]
auc <- mean(sample(pos.decision,1000,replace=T) > sample(neg.decision,1000,replace=T))
auc
## or
aucs <- replicate(2000,mean(sample(pos.decision,1000,replace=T) > sample(neg.decision,1000,replace=T)))
auc2 <- round(mean(aucs),4)
plot(FPR, TPR, col=4,lwd=5, type="l", main=paste("AUC=",auc2*100,"%"))
grid(5, 5, lwd = 1)
points(c(0,1), c(0,1), type="l", lty=2, lwd=2, col="grey")  
##
cut.op <- p0[which(TPR-FPR == max(TPR-FPR))]
cut.op

##calculate Accuracy,Precision,Recall,F1,MCC
TP<-sum(pos.decision>cut.op)

FN<-sum(pos.decision<cut.op)

FP<-sum(neg.decision>cut.op)

TN<-sum(neg.decision<cut.op)

Accuracy<-(TP+TN)/(TP+TN+FP+FN)

Precision<-TP/(TP+FP)

Recall<-TP/(TP+FN)

F1<-2*(Precision*Recall)/(Precision+Recall)

MCC<- (TP*TN-FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))

Accuracy
Precision
Recall
F1
MCC