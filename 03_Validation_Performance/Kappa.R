install.packages("psy")
library(psy)
data_kap <- read.csv("C:/Users/max22/Desktop/KAPPA.csv")
#二/多分类变量(2-years)
lkappa(data_kap[,c(4,5)])
#有序
#lkappa(data_kap[,c(4,5)]， type="weighted")
#计算95%CI
library(boot)
kappa.boot<-function(data,x){lkappa(data[x,])}
#kappa.boot<-function(data,x){lkappa(data[x,],type="weighted")}
res<-boot(data_kap[,c(4,5)],kappa.boot,1000)
quantile(res$t, c(0.025,0.975))

#二/多分类变量(5-years)
lkappa(data_kap[,c(6,7)])
#计算95%CI
library(boot)
kappa.boot<-function(data,x){lkappa(data[x,])}
#kappa.boot<-function(data,x){lkappa(data[x,],type="weighted")}
res1<-boot(data_kap[,c(6,7)],kappa.boot,1000)
quantile(res1$t, c(0.025,0.975))

#二/多分类变量(10-years)
lkappa(data_kap[,c(8,9)])
#计算95%CI
library(boot)
kappa.boot<-function(data,x){lkappa(data[x,])}
#kappa.boot<-function(data,x){lkappa(data[x,],type="weighted")}
res2<-boot(data_kap[,c(8,9)],kappa.boot,1000)
quantile(res2$t, c(0.025,0.975))
