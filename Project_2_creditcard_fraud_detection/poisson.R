

p1=creditcard[creditcard$Class==1,c(1,31)]
p1$Class <- as.character(p1$Class)
p1$Class <- as.numeric(p1$Class)
count=0
first=406
for (i in 1:nrow(p1))
{
  p1[i,'count'] <-count+p1[i,'Class']
  p1[i,'elapsed']<- p1[i,'Time']-first #+runif(1,0,0.01)
  first <- p1[i,'Time']
  count=count+1
  
}




plot(p1$Time)
fit <- glm( count ~ 1 + offset(log(Time)), data=p1, family=poisson)

install.packages('NHPoisson')
library(NHPoisson)
library(poisson)

scen = hpp.scenario(rate = 0.002847354 , num.events = 492, num.sims = 2)
plot(scen, main='My HPP Scenario')



lambda=1/60 #1 event per minute
time.span=60*60 #24 hours, with time granularity one second

aux<-simNHP.fun(rep(lambda,time.span))
out<-fitPP.fun(posE=aux$posNH,n=time.span,start=list(b0=0))
exp(coef(out)[1])
1/60
aux$posNH


out2<-fitPP.fun(posE=p2$Time,n=172792,start=list(b0=0))
lambda=exp(coef(out2)[1])
t=351
lambdat=lambda*t
n=1
(lambdat^n/factorial(n))*exp(-lambdat)


library(xlsx)
path <- paste("C:/users/",userid,"/desktop/p1.xlsx",sep = "")
p2<-read.xlsx(path,1)
ks.test(log10(p1$elapsed+0.5), "pnorm")
hist(log10(p1$elapsed+0.5))

set.seed(1234)
temp=creditcard[sample.int(nrow(var1_28),5000),]



shapiro.test(temp$V13)

p2$elapsed
Sys.setenv(LANG = "en")