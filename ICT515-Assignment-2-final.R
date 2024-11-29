# Install required packages
# Install required packages (if not already installed)
if (!require("pacman")) install.packages("pacman")
pacman::p_load(readxl, ggplot2, gridExtra, reshape2, rpart, rpart.plot, caret, pROC, randomForest, e1071, ipred)

dress_attribute_1 <- read_excel("Attribute-DataSet.xlsx")
names(dress_attribute_1)
dim(dress_attribute_1)

#check the datatype of each column 
str(dress_attribute_1)

#check the NA value in each column
colSums(is.na(dress_attribute_1))

#remove the NA values on each column 
dress_attribute_2 <- na.omit(dress_attribute_1)

#check the NA value in each column now to test if it's removed perfectly or not 

colSums(is.na(dress_attribute_2))
dim(dress_attribute_2)

View(dress_attribute_2)

#check each categorical columns unique values and sort it ascending to 
#descending to check if there any mispelling present in data or not 

lapply(dress_attribute_2[sapply(dress_attribute_2, is.character)], function(x) sort(unique(x), decreasing = TRUE))

#Theres lots of spelling error in each categorical column. I will correct the spelling 
# spelling in every categorical column 

#Style column "sexy" --> "Sexy" and all first letter get capitalized
dress_attribute_2$Style <- sapply(dress_attribute_2$Style, function(x) {
  if(tolower(x) == "sexy") {
    "Sexy"
  } else {
    paste(toupper(substring(x, 1, 1)), tolower(substring(x, 2)), sep = "")
  }
})

#Price column "low" will be "Low" ang "high" will be High
dress_attribute_2$Price <- gsub("^low$", "Low", dress_attribute_2$Price, ignore.case = FALSE)
dress_attribute_2$Price <- gsub("^high$", "High", dress_attribute_2$Price, ignore.case = FALSE)

dress_attribute_2$Price <- sapply(dress_attribute_2$Price, function(x) {
  paste(toupper(substring(x, 1, 1)), tolower(substring(x, 2)), sep = "")
})

#Size column "small" will be "S" and s will be "S" and free will be Free 

dress_attribute_2$Size <- sapply(dress_attribute_2$Size, function(x) {
  if (x == "small" || x == "s") {
    "S"
  } else if (x == "free") {
    "Free"
  } else {
    x
  }
})

#Season column "winter" will be "Winter", "summer" will be "Summer", "spring"
#will be "Spring" and "Automn" will be "Autumn"

dress_attribute_2$Season <- sapply(dress_attribute_2$Season, function(x) {
  if (x == "winter") {
    "Winter"
  } else if (x == "summer") {
    "Summer"
  } else if (x == "spring") {
    "Spring"
  } else if (x == "Automn") {
    "Autumn"
  } else {
    x
  }
})

# NeckLine column - "sweetheart" will  be "Sweetheart" , "NULL" will be"NA"
#and all other first letter will be capitalized

dress_attribute_2$NeckLine <- sapply(dress_attribute_2$NeckLine, function(x) {
  if (x == "sweetheart") {
    "Sweetheart"
  } else if (x == "NULL") {
    NA
  } else {
    paste(toupper(substring(x, 1, 1)), tolower(substring(x, 2)), sep = "")
  }
})

colSums(is.na(dress_attribute_2))

# SleeveLength column - "urndowncollor" will be "turndowncollor", "thressqatar"
# and threequater" will be threequarter, "sleveless", sleeevless and "sleevless" will 
# be "sleeveless", "halfsleeve" will be "half", "capsleeves" will be "cap-sleeves"
#and NULL will be NA 

dress_attribute_2$SleeveLength <- sapply(dress_attribute_2$SleeveLength, function(x) {
  if (x == "urndowncollor") {
    "Turndowncollor"
  } else if (x == "thressqatar" || x == "threequater") {
    "Threequarter"
  } else if (x == "sleveless" || x == "sleeevless" || x == "sleevless") {
    "Sleeveless"
  } else if (x == "halfsleeve") {
    "Half"
  } else if (x == "capsleeves") {
    "Cap-sleeves"
  } else if (x == "NULL") {
    NA
  } else {
    paste(toupper(substring(x, 1, 1)), tolower(substring(x, 2)), sep = "")
  }
})

#waiseline column "null" will be "NA" and capitalized the first letter
#of each entry


dress_attribute_2$waiseline <- sapply(dress_attribute_2$waiseline, function(x) {
  if (x == "null") {
    NA
  } else {
    # Capitalize the first letter of the entry
    paste(toupper(substring(x, 1, 1)), tolower(substring(x, 2)), sep = "")
  }
})

#Material column "sill" will be "silk", "modal" will be "model", "null" will 
#be "NA" and capitilize first letter of each entry

dress_attribute_2$Material <- sapply(dress_attribute_2$Material, function(x) {
  if (x == "sill") {
    "Silk"
  } else if (x == "modal") {
    "Model"
  } else if (x == "null") {
    NA
  } else {
    # Capitalize the first letter of each entry
    paste(toupper(substring(x, 1, 1)), tolower(substring(x, 2)), sep = "")
  }
})

#FabricType column "woolen" and "wollen" will be "woollen", "sattin" will bel
#"satin", "knitting" will be "knitted", "flannael" will be "flannel", "null" 
#will be "NA" and capitalise first letter of all entry 

dress_attribute_2$FabricType <- sapply(dress_attribute_2$FabricType, function(x) {
  if (x == "woolen" || x == "wollen") {
    "Woollen"
  } else if (x == "sattin") {
    "Satin"
  } else if (x == "knitting") {
    "Knitted"
  } else if (x == "flannael") {
    "Flannel"
  } else if (x == "null") {
    NA
  } else {
    # Capitalize the first letter of each entry
    paste(toupper(substring(x, 1, 1)), tolower(substring(x, 2)), sep = "")
  }
})

#Decoration column "null" and "none" will be "NA" 
#capitalize each word of every entry

dress_attribute_2$Decoration <- sapply(dress_attribute_2$Decoration, function(x) {
  if (x == "null" || x == "none") {
    NA
  } else {
    # Capitalize the first letter of the entry
    paste(toupper(substring(x, 1, 1)), tolower(substring(x, 2)), sep = "")
  }
})

#Patter Type column "leapord" will be "leopard", "null" will be "NA", "none"
#will be "NA" and Capitalize the first letter of each entry

dress_attribute_2$`Pattern Type` <- sapply(dress_attribute_2$`Pattern Type`, function(x) {
  if (x == "leapord") {
    "Leopard"
  } else if (x == "null" || x == "none") {
    NA
  } else {
    # Capitalize the first letter of the entry
    paste(toupper(substring(x, 1, 1)), tolower(substring(x, 2)), sep = "")
  }
})

colSums(is.na(dress_attribute_2))
#dim(dress_attribute_2)

#NA percentage of each column 
na_percentage <- colSums(is.na(dress_attribute_2)) / nrow(dress_attribute_2) * 100
na_percentage <- format(na_percentage, nsmall = 2, digits = 2)
na_percentage

#Price Distribution
ggplot(dress_attribute_2, aes(x = Price)) + 
  geom_bar(fill = "#0072B2") + 
  geom_text(stat='count', aes(label=..count..), vjust=-0.5) +
  labs(title = "Price Distribution", x = "Price", y = "Count") +
  theme_minimal()

ggplot(dress_attribute_2, aes(x = Size)) +
  geom_bar(fill = "#0072B2") +  
  geom_text(stat='count', aes(label=..count..), vjust=-0.5) +
  labs(title = "Size Distribution", x = "Size", y = "Count") +
  theme_minimal()

ggplot(dress_attribute_2, aes(x = Season)) +
  geom_bar(fill = "#0072B2") +  
  geom_text(stat='count', aes(label=..count..), vjust=-0.5) +
  labs(title = "Season Distribution", x = "Season", y = "Count") +
  theme_minimal()

# Heatmap of Size and Season distribution
contingency_table <- table(dress_attribute_2$Size, dress_attribute_2$Season)
contingency_melted <- melt(contingency_table)

ggplot(contingency_melted, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +  # Add white borders to cells
  geom_text(aes(label = value), color = "white") +
  scale_fill_gradient2(low = "lightblue", mid = "blue", high = "darkblue", midpoint = mean(contingency_melted$value)) +  # Diverging color scale
  labs(title = "Seasonal Popularity of Dress Sizes", x = "Size", y = "Season", fill = "Number of Dresses") +  # Improved title and labels
  theme_minimal() +
  scale_x_discrete(limits = c("S", "M", "L", "XL", "Free")) +  # Order sizes logically
  scale_y_discrete(limits = c("Spring", "Summer", "Autumn", "Winter"))  # Order seasons chronologically

# Heatmap for Size and Price distribution
size_price_table <- table(dress_attribute_2$Size, dress_attribute_2$Price)
size_price_melted <- melt(size_price_table)

# Create the heatmap
ggplot(size_price_melted, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +  # Add white borders to cells
  geom_text(aes(label = value), color = "white") +
  scale_fill_gradient2(low = "lightblue", mid = "blue", high = "darkblue", midpoint = mean(size_price_melted$value)) +  # Diverging color scale
  labs(title = "Price Variation Across Dress Sizes", x = "Size", y = "Price", fill = "Number of Dresses") +  # Improved title and labels
  theme_minimal() +
  scale_x_discrete(limits = c("S", "M", "L", "XL", "Free")) +  # Order sizes logically
  scale_y_discrete(limits = c("Low", "Average", "Medium", "High", "Very-high"))  # Order price categories

# Heatmap for Season and Price
season_price_table <- table(dress_attribute_2$Season, dress_attribute_2$Price)
season_price_melted <- melt(season_price_table)

# Create the heatmap
ggplot(season_price_melted, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +  # Add white borders to cells
  geom_text(aes(label = value), color = "white") +
  scale_fill_gradient2(low = "lightblue", mid = "blue", high = "darkblue", midpoint = mean(season_price_melted$value)) +  # Diverging color scale
  labs(title = "Seasonal Distribution of Dress Prices", x = "Season", y = "Price", fill = "Number of Dresses") +  # Improved title and labels
  theme_minimal() +
  scale_x_discrete(limits = c("Spring", "Summer", "Autumn", "Winter")) +  # Order seasons chronologically
  scale_y_discrete(limits = c("Low", "Average", "Medium", "High", "Very-high"))  # Order price categories

colSums(is.na(dress_attribute_2))

#Model Creation 

#convert 1 to Yes, 0 as No 
dress_attribute_2$Recommendation <- ifelse(dress_attribute_2$Recommendation == 1, "Yes", "No")

#
dress_attribute_3 <- dress_attribute_2[, c("Style", "Price", "Rating", "Size", "Season", "NeckLine", "SleeveLength", "Recommendation")]
colSums(is.na(dress_attribute_3))

#View(dress_attribute_2)
#drop NA 
dress_attribute_clean <- na.omit(dress_attribute_3)
colSums(is.na(dress_attribute_clean))
#dim(dress_attribute_2)
#dim(dress_attribute_clean)

tree_model <- rpart(Recommendation ~ ., data = dress_attribute_clean, method = "class")

# Plot the tree
rpart.plot(tree_model, type = 3, extra = 101, fallen.leaves = TRUE, main = "Classification Tree for Recommendation")

# Create the initial classification tree model with cross-validation
tree_model <- rpart(Recommendation ~ ., data = dress_attribute_clean, method = "class", cp = 0.01, xval = 10)

# Use the printcp function to display cross-validation results and find the optimal cp
printcp(tree_model)

# Prune the tree using the optimal cp value
optimal_cp <- tree_model$cptable[which.min(tree_model$cptable[,"xerror"]), "CP"]
pruned_tree <- prune(tree_model, cp = optimal_cp)

# Plot the pruned tree
rpart.plot(pruned_tree, type = 3, extra = 101, fallen.leaves = TRUE, main = "Pruned Classification Tree for Recommendation")

# Set the seed for reproducibility
set.seed(123)

# Define the split ratio
split_ratio <- 0.7

# Create an index for training data
train_index <- sample(seq_len(nrow(dress_attribute_clean)), size = split_ratio * nrow(dress_attribute_clean))

# Split the data into training and validation sets
training_set <- dress_attribute_clean[train_index, ]
validation_set <- dress_attribute_clean[-train_index, ]

# Make predictions on the validation set
predictions <- predict(pruned_tree, newdata = validation_set, type = "class")

predictions <- factor(predictions, levels = c("Yes", "No"))
validation_set$Recommendation <- factor(validation_set$Recommendation, levels = c("Yes", "No"))

# Now compute the confusion matrix
confusion_matrix <- confusionMatrix(predictions, validation_set$Recommendation)
print(confusion_matrix)

# Create the confusion matrix as a table
confusion_table <- table(Actual = validation_set$Recommendation, Predicted = predictions)
print(confusion_table)

# Get probabilities for the positive class "Yes"
probabilities <- predict(pruned_tree, newdata = validation_set, type = "prob")[, "Yes"]

# Create and plot the ROC curve
roc_curve <- roc(validation_set$Recommendation, probabilities)
plot(roc_curve, main = "ROC Curve for Validation Set")
auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))



#code for random forest 

# Set the seed for reproducibility
set.seed(123)

training_set$Recommendation <- as.factor(training_set$Recommendation)
validation_set$Recommendation <- as.factor(validation_set$Recommendation)

# Train the Random Forest model
rf_model <- randomForest(Recommendation ~ ., data = training_set, ntree = 100)

# Predict on the validation set
pred_rf <- predict(rf_model, newdata = validation_set)

# Ensure both predictions and actual values are factors with the same levels
pred_rf <- factor(pred_rf, levels = c("Yes", "No"))
validation_set$Recommendation <- factor(validation_set$Recommendation, levels = c("Yes", "No"))

# Compute the confusion matrix for Random Forest
confusion_matrix_rf <- confusionMatrix(pred_rf, validation_set$Recommendation)
print(confusion_matrix_rf)

# Display the confusion matrix as a table
confusion_table_rf <- table(Actual = validation_set$Recommendation, Predicted = pred_rf)
print(confusion_table_rf)

# Calculate ROC and AUC for the Random Forest model
# Get probabilities for the positive class "Yes"
rf_probabilities <- predict(rf_model, newdata = validation_set, type = "prob")[, "Yes"]

# Create and plot the ROC curve
roc_curve_rf <- roc(validation_set$Recommendation, rf_probabilities, levels = c("No", "Yes"))

# Plot the ROC curve
plot(roc_curve_rf, main = "ROC Curve for Random Forest Model")
auc_value_rf <- auc(roc_curve_rf)
print(paste("AUC for Random Forest:", auc_value_rf))

# Set up cross-validation and tuning grid
control <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation

# Define the tuning grid
tune_grid <- expand.grid(
  mtry = c(2, 3, 4),       # Number of variables randomly sampled at each split
  splitrule = "gini",      # Use the Gini index
  min.node.size = c(5, 10) # Minimum size of terminal nodes
)

#RF with KFOLD 

# Ensure that Recommendation is a factor
training_set$Recommendation <- as.factor(training_set$Recommendation)

# Set up cross-validation with 10 folds
control <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation

# Train the Random Forest model with cross-validation
set.seed(123)
rf_cv_model <- train(
  Recommendation ~ ., 
  data = training_set, 
  method = "rf",             
  trControl = control,        
  tuneLength = 5              
)

# Display the results of cross-validation
print(rf_cv_model)

# Check the best model parameters
print(rf_cv_model$bestTune)

#display the accuracy metrics from each fold
accuracy_results <- rf_cv_model$results
print(accuracy_results)

# Best accuracy obtained on cross-validation
best_accuracy <- max(accuracy_results$Accuracy)
print(paste("Best Cross-Validated Accuracy:", best_accuracy))

#Model Naive Bayes 

# Train the Naive Bayes model on the training set
nb_model <- naiveBayes(Recommendation ~ ., data = training_set)
# Predict on the validation set
predictions_nb <- predict(nb_model, newdata = validation_set)

# Confusion Matrix and Accuracy
confusion_matrix_nb <- confusionMatrix(predictions_nb, validation_set$Recommendation)
print(confusion_matrix_nb)

confusion_table <- table(Actual = validation_set$Recommendation, Predicted = predictions_nb)
print(confusion_table)

accuracy <- sum(diag(confusion_table)) / sum(confusion_table)
accuracy

# Get probabilities for the positive class "Yes"
nb_probabilities <- predict(nb_model, newdata = validation_set, type = "raw")[, "Yes"]

# Create and plot the ROC curve
roc_curve_nb <- roc(validation_set$Recommendation, nb_probabilities, levels = c("No", "Yes"))
plot(roc_curve_nb, main = "ROC Curve for Naive Bayes Model")
auc_value_nb <- auc(roc_curve_nb)
print(paste("AUC for Naive Bayes:", auc_value_nb))

#applied bagging 

# --- Model Evaluation ---
cm_dt <- as.data.frame(confusion_matrix$table)
cm_rf <- as.data.frame(confusion_matrix_rf$table)
cm_nb <- as.data.frame(confusion_matrix_nb$table)

p1 <- ggplot(cm_dt, aes(Prediction, Reference, fill = Freq)) + 
  geom_tile() +
  geom_text(aes(label = sprintf("%d", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Decision Tree Confusion Matrix") +
  theme_minimal()

p2 <- ggplot(cm_rf, aes(Prediction, Reference, fill = Freq)) + 
  geom_tile() +
  geom_text(aes(label = sprintf("%d", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Random Forest Confusion Matrix") +
  theme_minimal()

p3 <- ggplot(cm_nb, aes(Prediction, Reference, fill = Freq)) + 
  geom_tile() +
  geom_text(aes(label = sprintf("%d", Freq)), vjust = 1) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Naive Bayes Confusion Matrix") +
  theme_minimal()

grid.arrange(p1, p2, p3, ncol = 3)

# Define a wrapper function for Naive Bayes to use with bagging
nb_bagging <- function(data, indices) {
  boot_data <- data[indices, ]
  model <- naiveBayes(Recommendation ~ ., data = boot_data, laplace = best_laplace)
  return(model)
}

# Define the function to calculate metrics
calculate_metrics <- function(conf_matrix) {
  # Extract the confusion table from the confusionMatrix object
  cm <- conf_matrix$table
  
  # Validate table dimensions and classes
  if (!all(dim(cm) == c(2,2)) || 
      !all(rownames(cm) %in% c("Yes", "No")) || 
      !all(colnames(cm) %in% c("Yes", "No"))) {
    stop("Confusion matrix must be 2x2 with 'Yes'/'No' labels")
  }
  
  # Define TP, FN, FP, and TN based on the table
  tp <- cm["Yes", "Yes"]
  fn <- cm["Yes", "No"]
  fp <- cm["No", "Yes"]
  tn <- cm["No", "No"]
  
  # Calculate metrics with safe division
  safe_divide <- function(n, d) ifelse(d > 0, n/d, NA)
  
  accuracy <- (tp + tn) / sum(cm)
  precision <- safe_divide(tp, tp + fp)
  recall <- safe_divide(tp, tp + fn)  # Also known as Sensitivity
  specificity <- safe_divide(tn, tn + fp)
  
  # Calculate F1 score only if both precision and recall are valid
  f1_score <- if (!is.na(precision) && !is.na(recall) && (precision + recall) > 0) {
    2 * ((precision * recall) / (precision + recall))
  } else {
    NA
  }
  
  balanced_accuracy <- (recall + specificity) / 2
  
  # Return metrics as a data frame
  return(data.frame(
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    Specificity = specificity,
    F1_Score = f1_score,
    Balanced_Accuracy = balanced_accuracy
  ))
}


# Calculate metrics for each model
metrics_conf_matrix <- calculate_metrics(confusion_matrix)
metrics_conf_matrix_nb <- calculate_metrics(confusion_matrix_nb)
metrics_conf_matrix_rf <- calculate_metrics(confusion_matrix_rf)

# Combine the metrics into a single data frame for plotting
all_metrics <- rbind(
  cbind(Model = "Decision Tree", metrics_conf_matrix),
  cbind(Model = "Naive Bayes", metrics_conf_matrix_nb),
  cbind(Model = "Random Forest", metrics_conf_matrix_rf)
)

# Plot the metrics

# Melt the data for ggplot2
all_metrics_long <- melt(all_metrics, id.vars = "Model")

# Barchart for performance matrices comparision
ggplot(all_metrics_long, aes(x = Model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = round(value, 2)), 
            position = position_dodge(width = 0.9), 
            vjust = -0.3, size = 3) +
  labs(title = "Comparison of Model Performance Metrics",
       x = "Model",
       y = "Metric Value") +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal()

