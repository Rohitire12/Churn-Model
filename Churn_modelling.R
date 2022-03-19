# ANN - Artificial Neural Network
#Setting the library
setwd("~/Desktop")
# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')

str(dataset)
dataset = dataset[4:14]

# Encoding the categorical variables as factors
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])

# Fitting ANN to the Training set
# install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)
ANN_model = h2o.deeplearning(y = 'Exited',
                         training_frame = as.h2o(training_set),
                         activation = 'Rectifier',
                         hidden = c(5,5),
                         epochs = 100,
                         train_samples_per_iteration = -2)

# Predicting the Test set results
predict = h2o.predict(ANN_model, newdata = as.h2o(test_set[-11]))
predict = (predict > 0.5)
predict = as.vector(predict)

# Making the Confusion Matrix
conf_matrix = table(test_set[, 11], predict)
conf_matrix
ANN_model_Accrucy = mean(predict == test_set$Exited)
cat("\nEvaluation of the Model\n")
ANN_model_Accrucy
#plot(ANN_model)
#ANN_model
h2o.shutdown()
ANN_model


