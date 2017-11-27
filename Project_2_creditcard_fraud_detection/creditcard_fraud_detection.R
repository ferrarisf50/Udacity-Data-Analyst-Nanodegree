userid <- Sys.getenv("USERNAME")
setwd(paste("C:/users/",userid,"/desktop/creditcard",sep = ""))


creditcard=read.csv("creditcard.csv")

creditcard$Class=factor(creditcard$Class)

## remove duplicate rows
#table1 <- unique(creditcard)
#table2 <- creditcard[duplicated(creditcard),]
library(ggplot2)
qplot(x=Class,data=creditcard)

# first=0
# for (i in 1:nrow(creditcard))
# {
#   
#   creditcard[i,'elapsed']<- creditcard[i,'Time']-first
#   first <- creditcard[i,'Time']
#   print(i)
# }
# hist(log10(creditcard$elapsed+2))


#histogram plot for Class.

qplot(x=Class,data=table1)

summary(table1$Class)
table(table1$Class)
#Only 0.001% transactions are fraudulent.
#print statistics;

summary(table1)
#plot Time
#by minute

qplot(x=Time,data=creditcard[creditcard$Class==1,],geom='histogram',color="#F8766D", binwidth=60)+
  labs(x="Time (Class=1)")
#
qplot(x=Time,data=creditcard,geom='freqpoly')

ggplot(creditcard,aes(Time,V4))+geom_smooth()

ggplot(aes(x=Time),data=creditcard)+
  geom_density(aes(y=..density..,color=Class,fill=Class),alpha=0.2)+
  scale_fill_manual(values = c("#00BFC4","#F8766D"))+
  scale_color_manual(values = c("#00BFC4","#F8766D"))

a <- as.POSIXct(4566,origin = "1960-01-01",tz = "GMT")
aa <-as.numeric(strftime(a, format="%H",tz = "GMT"))

creditcard$Hour <- as.numeric(strftime(as.POSIXct(creditcard$Time,origin = "1960-01-01",tz = "GMT"), format="%H",tz = "GMT"))

#histogram plot for Amount.
qplot(x=log10(Amount+1),data=creditcard)
qplot(x=scale(Amount),data=creditcard)
qplot(x=Amount,data=creditcard,geom='freqpoly')

creditcard$Amount_A=log10(creditcard$Amount+1)

qplot(x=Amount_A,data=creditcard)+
  geom_histogram(bins=10000)+
facet_grid(Class ~ .)

ggplot(aes(x=log10(Amount+1)),data=creditcard)+
  geom_density(aes(y=..density..,color=Class,fill=Class),alpha=0.2)+
  scale_fill_manual(values = c("#00BFC4","#F8766D"))+
  scale_color_manual(values = c("#00BFC4","#F8766D"))

# scatterplot amount vs time vs class
ggplot(creditcard,aes(Time,Amount))+geom_point(aes(color=Class,size=Class,alpha=Class))+
  scale_color_manual(values =  c("white","red"))+
  scale_size_manual(values=c(0.001,1))+
  scale_alpha_manual(values=c(0.1,1))+
  scale_y_continuous(limits = c(0, 5000))+
  theme(panel.background = element_rect(fill = 'black'),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank())


ggplot(creditcard,aes(Time,Amount_A))+geom_point(aes(color=Class,size=Class,alpha=Class))+
  scale_color_manual(values =  c("white","red"))+
  scale_size_manual(values=c(0.001,1))+
  scale_alpha_manual(values=c(0.1,1))+
  labs(y="log10(Amount+1)")+
  theme(panel.background = element_rect(fill = 'black'),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank())

# V1-V28 Vs Time scatterplot

a <-list()
for (i in 2:29)
{
a[[i]] <- ggplot(creditcard,aes_string(x='Time',y=names(creditcard)[i]))+geom_jitter(aes(color=Class,size=Class,alpha=Class))+
  scale_color_manual(values =  c("white","red"))+
  scale_size_manual(values=c(1,1))+
  scale_alpha_manual(values=c(1,1))+
  theme(panel.background = element_rect(fill = 'black'),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank())
  print(a[[i]])
} 

#pair plot for v1-v28 and Class

library(GGally)
table4=creditcard[creditcard$Class==0,]

set.seed(1234)
names(table4)
temp <-table4[sample.int(nrow(table4),10000),]
temp2 <- rbind(temp,creditcard[creditcard$Class==1,])
#ggpairs(temp2)


ggpairs(table1[,c(2,3,31)],columns=1:2, 
    mapping = ggplot2::aes(color = Class), 
  
    lower = list(continuous = wrap("points", alpha = 0.3)),
    diag = list(continuous = wrap("densityDiag", alpha = 0.3)))

##full graph
p<-ggpairs(temp2,columns=2:30, 
        mapping = ggplot2::aes(color = Class), 
        
        lower = list(continuous = wrap("points", alpha = 0.3)),
        diag = list(continuous = wrap("densityDiag", alpha = 0.3)))

for(i in 1:p$nrow) {
  for(j in 1:p$ncol){
    p[i,j] <- p[i,j] + 
      scale_fill_manual(values=c("#00BFC4","#F8766D")) +
      scale_color_manual(values=c("#00BFC4","#F8766D"))  
  }
}      
print(p) 

#qplot(x=Amount,data=creditcard,binwidth=50)+
#  scale_x_continuous(limits = c(0, 1000), breaks = seq(0, 1000, 100))+
#  facet_grid(Class ~ .)
memory.limit()

##plot V1-V28

V1_V28_long <- melt(creditcard[,2:29])
head(V1_V28_long)

a <-list()
for (i in 2:29)
{
  a[[i]]=ggplot(aes_string(x=names(creditcard)[i]),data=creditcard)+
    geom_histogram(bins=1000)
  print(a[[i]])
} 


creditcard$V1_A=log10(-creditcard$V1+3)

ggplot(aes(x=(V28+16)^(1/3)),data=creditcard)+
  geom_histogram(bins=1000)

ggplot(aes(x=(1/creditcard$V28)),data=creditcard)+
  geom_histogram(bins=1000)

library(moments)
skewness(log10(creditcard$V28+24))

ggplot(aes(x=log10(-creditcard$V1+3)),data=creditcard)+
  geom_density(aes(y=..density..,color=Class,fill=Class),alpha=0.2)+
  scale_fill_manual(values = c("#00BFC4","#F8766D"))+
  scale_color_manual(values = c("#00BFC4","#F8766D"))+
  labs(x='log10(-V1+3')

for (i in 2:33)
{
  print(i)
  print(skewness(creditcard[,i]))
}

a <-list()
for (i in 2:29)
{
  a[[i]]=ggplot(aes_string(x=names(creditcard)[i]),data=creditcard)+
    geom_density(aes(y=..density..,color=Class,fill=Class),alpha=0.2)+
    scale_fill_manual(values = c("#00BFC4","#F8766D"))+
    scale_color_manual(values = c("#00BFC4","#F8766D"))
  
   print(a[[i]])
} 


multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}
multiplot(plotlist=a,cols=4)




##plot density of Amount
ggplot(aes(x=log10(Amount+1)),data=table1)+
  geom_density(aes(y=..density..,color=Class,fill=Class),alpha=0.2)


aaa=names(table1)[1]
aaa

ggpairs(table1,columns=2, 
          mapping = ggplot2::aes(color = Class))
        
  
  

qplot(x=V1,data=table1)+ 
scale_x_sqrt()

qplot(x=sqrt(V1),data=creditcard)


temp3=temp2
temp3$V1 <- sqrt(temp3$V1^(1/3))

ggpairs(temp3,columns=1, 
        mapping = ggplot2::aes(color = Class), 
        
        lower = list(continuous = wrap("points", alpha = 0.3)),
        diag = list(continuous = wrap("densityDiag", alpha = 0.3)))


##autoplot(prcomp(df), data = creditcard, colour = 'Class',xlim=c(-0.025,0))+
##  scale_color_manual(values = c(NA,"black"))

##ggparallel
var1_28=creditcard[creditcard$Class==0,c(2:31)]
set.seed(1234)
temp <-var1_28[sample.int(nrow(var1_28),1000),]
temp2 <- rbind(temp,creditcard[creditcard$Class==1,c(2:31)])


ggparcoord(data=temp2,columns=c(2:30),  groupColumn="Class",
           scale = "std",
           alphaLines=0.3)
           


# V1-V28 engineering

set.seed(1234)
temp <-table1[sample.int(nrow(table1),5000),]
shapiro.test(temp$V14)

## V1-V28 box plot
melted_V1_V28 <- melt(creditcard[,1:29],id.vars="Time")
head(melted_V1_V28)

ggplot(melted_V1_V28) +
  geom_boxplot(aes(x=variable, y=value, color=variable))+
  scale_y_continuous(limits = c(-10, 10))
  

ggplot(melted_V1_V28) +
  geom_violin(aes(x=variable, y=value, color=variable))+
  scale_y_continuous(limits = c(-50, 50))


## correlation heat map
head(table1)
table3<-unique(creditcard)

table4 <- cor(creditcard[creditcard$Class==1,c(2:30,33,34)])
head(table4)
library(reshape2)
melted_table4 <- melt(table4)


ggplot(data = melted_table4, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile(color="white")+
  scale_fill_gradient2(low = "blue", high = "blue", mid = "white")

ggplot(data = melt(cor(table3[,2:31])), aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile(color="white")+
  scale_fill_gradient2(low = "blue", high = "blue", mid = "white")


### high dimension plot

library(Rtsne)

#remove v8 v13 v15 v20 v22 v23 v24 v25 v26 v27 v28 Amount?
remove <- c("Time","V8","V13","V15","V20","V22","V23","V24","V25","V26","V27","V28","Amount")

remove <- c("Time","V8","V13","V15","V20","V22","V23","V24","V25","V26","V27","V28","Amount")
remove <- c("Time","V8","V13","V15","V20","V22","V23","V24","V25","V26")
remove <- c("Time","V8","V13","V15","V20","V22","V23","V24","V25","V26","V27","V28")

#new features after carefully exploring
remove <- c("Time","V1","V6","V8","V13","V15","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Amount_A")

table5=unique(creditcard[ , -which(names(creditcard) %in% remove)])
table5$Hour <- as.factor(table5$Hour)
summary(table5$Hour)

# table4=unique(table5[,-14])

# table5$V1 <- log(table5$V1+60)
# table5$Amount <- log(table5$Amount+2)

set.seed(456)
temp=table5[table5$Class==0,]
temp <-temp[sample.int(nrow(temp),10000),]

temp2 <- rbind(temp,table5[table5$Class==1,])





tsne <- Rtsne(temp2, dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
#exeTimeTsne<- system.time(Rtsne(temp2, dims = 3, perplexity=30, verbose=TRUE, max_iter = 1000))
temp3<-as.data.frame(tsne$Y)




ggplot(temp3,aes(V1,V2))+geom_point(aes(color=temp2$Class,size=temp2$Class,alpha=temp2$Class))+
  scale_color_manual(values =  c("#00BFC4","#F8766D"))+
  scale_size_manual(values=c(1,1))+
  scale_alpha_manual(values=c(0.1,1))



#no sampling:
table5=unique(creditcard[ , -which(names(creditcard) %in% remove)])
table5$Hour <- as.factor(table5$Hour)

tsne1 <- Rtsne(table5, dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)

temp3<-as.data.frame(tsne1$Y)

ggplot(temp3,aes(V1,V2))+geom_point(aes(color=temp2$Class,size=temp2$Class,alpha=temp2$Class))+
  scale_color_manual(values =  c("#00BFC4","#F8766D"))+
  scale_size_manual(values=c(1,1))+
  scale_alpha_manual(values=c(0.01,1))

#explore kurtosis

a<-list()
b<-list()
for (i in 2:29)
{
  diff<-abs(median(creditcard[creditcard$Class==0,i])-median(creditcard[creditcard$Class==1,i]))
  name <- names(creditcard)[i]
  kurtosis <- kurtosis(creditcard[creditcard$Class==1,i])
  sprintf("%s : %f",name,diff)
  b<- c(b,kurtosis)
  a <- c(a,diff)
}

ggplot(aes(x=as.factor(Hour), y=V4),data=creditcard) +
  geom_boxplot()

ggplot(data=melted_V1_V28,aes(x=Time,y=value,color=variable))+
  geom_smooth()

ggplot(aes(x=Time, y=V1,alpha=0.1),data=creditcard) +
  geom_point()

summary(creditcard[creditcard$Class==1,30])

### TIme vs V3 vs Hour

ggplot(creditcard,aes(Time,V12))+geom_point(aes(color=Hour))


##  V1-V28 correlation vs Hour

ggplot(creditcard,aes(V18,V17))+geom_point(aes(color=Time))

  
ggplot(creditcard,aes(V1,Amount))+geom_point(aes(color=Class,size=Class,alpha=Class))+
  scale_color_manual(values =  c("white","red"))+
  scale_size_manual(values=c(1,1))+
  scale_alpha_manual(values=c(0.1,1))+
  scale_y_continuous(limits = c(0, 5000))+
  theme(panel.background = element_rect(fill = 'black'),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank())