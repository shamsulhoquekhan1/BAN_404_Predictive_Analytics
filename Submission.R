Churn <- read.csv("Churn.csv")
Churn


#testing correlation between independant variables
correlation <- cor(Churn$download_avg, Churn$upload_avg)
print(correlation)



cross_table <- table(Churn$is_tv_subscriber, Churn$is_movie_package_subscriber)
cramer_v <- sqrt(chisq.test(cross_table)$statistic / (sum(cross_table) * (min(dim(cross_table)) - 1)))

# Print Cram?r's V coefficient
print(cramer_v)

#droping columns
Churn <- subset(Churn, select = -c(upload_avg, is_movie_package_subscriber, id))
Churn





# Set the seed for reproducibility
set.seed(65923764)

# Number of rows in the data
num_rows <- nrow(Churn)

# Generate a random permutation of row indices
perm <- sample(num_rows)

# Calculate the number of rows for training and test datasets
num_train <- floor(num_rows / 2)
num_test <- num_rows - num_train

# Create the training dataset
train_data <- Churn[perm[1:num_train], ]

# Create the test dataset
test_data <- Churn[perm[(num_train + 1):num_rows], ]


#(b)
#finding useful predictors for avg bill
summary(Churn$bill_avg)

correlation_matrix <- cor(Churn[, c("bill_avg", "is_tv_subscriber", "subscription_age", "remaining_contract", "service_failure_count", "download_avg", "download_over_limit", "churn")])

# Display the correlation matrix
print(correlation_matrix)

plot(Churn$download_avg, Churn$bill_avg, xlab = "Download Average (GB)", ylab = "Bill Average")
boxplot(bill_avg ~ is_tv_subscriber, data = Churn, xlab = "download_over_limit", ylab = "Bill Average")


#(c)
#producing model to predcit avg bill 
model <- lm(bill_avg ~ download_avg + download_over_limit + service_failure_count, data = train_data)
predictions <- predict(model, newdata = test_data)  # Generate predictions on the test dataset
actual_values <- test_data$bill_avg  # Extract the actual values from the test dataset

# Calculate evaluation metrics
rmse <- sqrt(mean((predictions - actual_values)^2))  # Root Mean Squared Error
mae <- mean(abs(predictions - actual_values))  # Mean Absolute Error
r_squared <- summary(model)$r.squared  # R-squared value

# Print the evaluation metrics
cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")
cat("R-squared:", r_squared, "\n")





#(d)
#Lasso model
library(glmnet)

# Convert the categorical variables to factors if needed
train_data$is_tv_subscriber <- as.factor(train_data$is_tv_subscriber)
train_data$churn <- as.factor(train_data$churn)

# Prepare the predictors and response variables
predictors <- model.matrix(~.-bill_avg, data = train_data)[,-1]  # Remove the intercept column
response <- train_data$bill_avg

# Perform LASSO regression with cross-validation
lasso_model <- cv.glmnet(predictors, response, alpha = 1, standardize = TRUE, nfolds = 5)

# Identify the optimal value of lambda (regularization parameter)
optimal_lambda <- lasso_model$lambda.min

print(optimal_lambda)

# Refit the LASSO model with the optimal lambda
lasso_model_optimal <- glmnet(predictors, response, alpha = 1, standardize = TRUE, lambda = optimal_lambda)

# Display the coefficients of the optimal LASSO model
lasso_coeffs_optimal <- coef(lasso_model_optimal)
print(lasso_coeffs_optimal)







#comparison
ols_model <- lm(bill_avg ~ ., data = train_data)

# Extract the coefficient estimates from the OLS model
ols_coeffs <- coef(ols_model)

# Extract the variable names (predictor names)
predictor_names <- names(ols_coeffs)

# Create a dataframe to store the coefficients for comparison
coeff_comparison <- data.frame(LASSO_Coefficients = lasso_coeffs_optimal[, "s0"], OLS_Coefficients = ols_coeffs)

# Print the coefficient comparison table
print(coeff_comparison)






#(e)
#Evaluating predictions
# Convert the categorical variables to factors if needed
test_data$is_tv_subscriber <- as.factor(test_data$is_tv_subscriber)
test_data$churn <- as.factor(test_data$churn)

# Prepare the predictors and response variables
predictors <- model.matrix(~.-bill_avg, data = test_data)[,-1]  # Remove the intercept column
response <- test_data$bill_avg



# Compute predictions using the LASSO model
lasso_predictions <- predict(lasso_model_optimal, newx = predictors)

# Calculate the evaluation metric (e.g., Root Mean Squared Error)
rmse <- sqrt(mean((lasso_predictions - response)^2))
r_squared <- 1 - sum((response - lasso_predictions)^2) / sum((response - mean(response))^2)

# Print the evaluation metric
cat("RMSE:", rmse, "\n")
cat("R-squared:", r_squared, "\n")




#(f)
#regression_tree
library(tree)


# Fit a regression tree model
tree_model <- tree(bill_avg~.,data= train_data)

# We will use cross validation to find optimal level of tree complexity.


cv.tree =cv.tree(tree_model,FUN=prune.tree)
plot(cv.tree$size,cv.tree$dev,type='b')

# Looking at the plot, I believe having tree size 8 gives the lowest prediction error.

pruned.tree <- prune.tree(tree_model,best=8)

plot(pruned.tree)
text(pruned.tree)


# The plot show the first split comes from download_avg (smaller than 422.45 goes left)
# One the left, next split is download_over_limit < 0.5. We have a prediction if
# it is more than 0.5 and, if not, next splits comes from download_avg, and then 
# subscription_age and is_tv_subscriber. The right side of download_avg split has
# split at download_avg, right side prediction and left side another split from 
# remaining_contract.

# G)------

# We are fitting the pruned.tree in the new test data set to predict.

pred_tree <- predict(pruned.tree, newdata=test_data)

# We ill evaluate the model through calculating MSE

testMSE_tree <- mean((test_data$bill_avg - pred_tree)^2)
testMSE_tree

# The MSE of tree model is 134.7101, which is higher the simple OLS model's 
# OLS MSE 125.163. So, tree model is not doing well at predicting.

# (H)


#random_forrest
library(randomForest)

# In random forests, we take subset of predictors. So, we will pick 3 predictors
# at a time for each time.

# Fit a random forest model
rf_model <- randomForest(bill_avg ~ ., data = train_data, ntree = 50, mtry=3)

# Plot variable importance
varImpPlot(rf_model)


#The variable importance plot generated from the random forest model provides 
#insights into the importance of each predictor variable in predicting the target 
#variable, "bill_avg". It ranks the variables based on their contribution to the
#model's performance.

#In particular, the variable importance plot includes a measure called 
#"IncNodePurity" or "MeanDecreaseGini" as an indicator of variable importance. 
#This measure represents the average improvement in the model's node purity 
#(impurity reduction) achieved by each predictor variable across all the trees 
#in the random forest.

#Higher values of IncNodePurity indicate that the variable has a stronger 
#influence on the model's predictions. It signifies that the variable plays a 
#more significant role in reducing the impurity (e.g., Gini index) and improving
#the accuracy of the model.




# (i)

rf_predictions <- predict(rf_model, newdata = test_data)

# Extract the actual values from the test dataset
actual_values <- test_data$bill_avg

# Calculate evaluation metrics
rmse <- sqrt(mean((rf_predictions - actual_values)^2))  # Root Mean Squared Error
mae <- mean(abs(rf_predictions - actual_values))  # Mean Absolute Error
r_squared <- cor(actual_values, rf_predictions)^2  # R-squared value

# Print the evaluation metrics
cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")
cat("R-squared:", r_squared, "\n")









# Task 2

#a)

# Load the dataset
Churn <- read.csv("churn.csv")
# Check the structure of the dataset
str(Churn)
# We can remove the 'id' variable as it does not contribute to the prediction of churn.
Churn$id <- NULL
# Convert variables to appropriate data types
Churn$bill_avg <- as.numeric(Churn$bill_avg)
Churn$is_tv_subscriber <- as.factor(Churn$is_tv_subscriber)
Churn$is_movie_package_subscriber <- as.factor(Churn$is_movie_package_subscriber)
# Handle missing values if present 
# Check for missing values in the dataset
colSums(is.na(Churn))
# Summary statistics for numerical variables 
summary(Churn[c("subscription_age", "bill_avg", "remaining_contract", "service_failure_count", "download_avg", "upload_avg", "download_over_limit")])



# Boxplots of numerical variables against churn
library(ggplot2)
ggplot(Churn, aes(x= churn, y= subscription_age)) + geom_boxplot() + labs(x= "Churn", y= "Subscription Age")
ggplot(Churn, aes(x= churn, y= bill_avg)) + geom_boxplot() + labs(x= "Churn", y= "Bill Average")
ggplot(Churn, aes(x= churn, y= remaining_contract)) + geom_boxplot() + labs(x= "Churn", y= "Remaining COntract")
ggplot(Churn, aes(x= churn, y= service_failure_count)) + geom_boxplot() + labs(x= "Churn", y= "Service Failure Count")
ggplot(Churn, aes(x= churn, y= download_avg)) + geom_boxplot() + labs(x= "Churn", y= "Download Average")
ggplot(Churn, aes(x= churn, y= upload_avg)) + geom_boxplot() + labs(x= "Churn", y= "Upload Average")
ggplot(Churn, aes(x= churn, y= download_over_limit)) + geom_boxplot() + labs(x= "Churn", y= "Download Over Limit")

#b)

# First 50 observations of the churn variable
churn_data <- Churn$churn[1:50]
#Calculate the sample fraction of churners
p_hat <- mean(churn_data)
p_hat
#Number of bootstrap iterations
n_bootstrap <- 71893
# Initialize vector to store bootstrap estimates
bootstrap_estimates <- numeric(n_bootstrap)
# Perform bootstrap
for (i in 1:n_bootstrap) {
  # Create bootstrap sample
  bootstrap_sample <- sample(churn_data, replace = TRUE)
  
  # Calculate sample fraction of churners in the bootstrap sample
  bootstrap_estimates[i] <- mean(bootstrap_sample)
}

# Sort bootstrap estimates in ascending order
sorted_estimates <- sort(bootstrap_estimates)

# Calculate the 95% confidence interval using bootstrap
lower <- sorted_estimates[round(n_bootstrap * 0.025)]
upper <- sorted_estimates[round(n_bootstrap * 0.975)]

# Calculate the standard approximation confidence interval
standard_lower <- p_hat - 1.96 * sqrt(p_hat * (1 - p_hat) / length(churn_data))
standard_upper <- p_hat + 1.96 * sqrt(p_hat * (1 - p_hat) / length(churn_data))

# Print the confidence intervals
cat("Bootstrap Confidence Interval: [", lower, ",", upper, "]\n")
cat("Standard Approximation Confidence Interval: [", standard_lower, ",", standard_upper, "]\n")

#c)

# Fit logistic regression model
model <- glm(churn ~ ., data = Churn, family = "binomial")

# Print the model summary
summary(model)

#d)

# Obtain predicted probabilities for churn
predicted_probs <- predict(model, newdata = Churn, type = "response")

# Convert predicted probabilities to binary predictions (0 or 1)
predicted_churn <- ifelse(predicted_probs >= 0.5, 1, 0)

# Evaluate predictions
confusion_matrix <- table(Churn$churn, predicted_churn)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
precision <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
recall <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print evaluation metrics
cat("Confusion Matrix:\n")
print(confusion_matrix)
cat("\nAccuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")

#e)

library(randomForest)

# Set the number of trees in the random forest (adjust if needed)
n_trees <- 100
# Convert churn variable to a factor
Churn$churn <- factor(Churn$churn)

# Fit the random forest model
model_rf <- randomForest(churn ~ ., data = Churn, mtry = 3, ntree = n_trees)

# Obtain predicted probabilities for churn
predicted_probs_rf <- predict(model_rf, newdata = Churn, type = "prob")[, 2]

# Convert predicted probabilities to binary predictions (0 or 1)
predicted_churn_rf <- ifelse(predicted_probs_rf >= 0.5, 1, 0)

# Evaluate predictions
confusion_matrix_rf <- table(Churn$churn, predicted_churn_rf)
accuracy_rf <- sum(diag(confusion_matrix_rf)) / sum(confusion_matrix_rf)
precision_rf <- confusion_matrix_rf[2, 2] / sum(confusion_matrix_rf[, 2])
recall_rf <- confusion_matrix_rf[2, 2] / sum(confusion_matrix_rf[2, ])
f1_score_rf <- 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)

# Print evaluation metrics
cat("Confusion Matrix (Random Forest):\n")
print(confusion_matrix_rf)
cat("\nAccuracy (Random Forest):", accuracy_rf, "\n")
cat("Precision (Random Forest):", precision_rf, "\n")
cat("Recall (Random Forest):", recall_rf, "\n")
cat("F1 Score (Random Forest):", f1_score_rf, "\n")
