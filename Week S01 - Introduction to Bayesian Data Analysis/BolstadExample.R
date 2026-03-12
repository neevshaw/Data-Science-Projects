# Setup libraries
library(Bolstad)

# Plot of Beta (1,1) prior
a<- 1
b<- 1
prob.p <-seq(0,1,.1)
prior.d <-dbeta(prob.p, a, b)
plot(prob.p, prior.d, 
     type="l", 
     main = "Prior Density for P",
     xlab = "Proportion",
     ylab = "Density")

# 8 successes observed in 20 trials with a Beta(1, 1) prior
results <- binobp(8, 20, 1, 1, plot = TRUE) 
par(mfrow = c(3, 1))
y.lims=c(0, 1.1*max(results$posterior, results$prior))
plot(results$pi, results$prior, ylim=y.lims, 
     type="l", 
     xlab=expression(pi), 
     ylab="Density", 
     main="Prior")
polygon(results$pi, results$prior, col="red")

plot(results$pi, 
     results$likelihood, 
     ylim=c(0,0.25), 
     type="l", 
     xlab=expression(pi), 
     ylab="Density", 
     main="Likelihood")
polygon(results$pi, results$likelihood, col="green")

plot(results$pi, 
     results$posterior, 
     ylim=y.lims, 
     type="l", 
     xlab=expression(pi), 
     ylab="Density", 
     main="Posterior")
polygon(results$pi, results$posterior, col="blue")
par(mfrow = c(1, 1))