#pca test


library(MASS)
u1=c(0,2)
sigma1=matrix(c(2,0,0,2),2,2)
set.seed(1234)
table1=mvrnorm(1000,mu=u1,Sigma=sigma1)
plot(table1)

table1.pca1= prcomp(table1,center=F,scale=F)
plot(table1.pca1$x)

plot(table1.pca2)
