a <- read.table("C:/Users/max22/Desktop/N_H_e_Validation1.txt")
a <- as.matrix(a)

label <- a[,2]
decision <- a[,1]

ngrids <- 100
c<-ngrids-1
P <- rep(0, ngrids)
R <- rep(0, ngrids)
p0 <- rep(0, ngrids)
A <- rep(0, ngrids)

for(i in 0:ngrids)
{
  p0[i] <- i/ngrids
  pred_label <- 1*(decision > p0[i])
  R[i] <- sum(pred_label * label) / sum(label)
  P[i] <- sum(pred_label * label) / sum(pred_label)
  A[i] <- sum((pred_label == label)*1)/nrow(a)
}


plot(R, P, col=4,lwd=5, type="l",xlab="Recall",ylab="Precision", main="PR Curve")
grid(5, 5, lwd = 1)

accuracy <- max(A)
accuracy

