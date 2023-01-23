# Neural Network Charity Analysis

## Purpose
The purpose of this analysis was to help Alphabet Soup develop a neural network model to predict which non-profit organizations they should donate to. To do this, data on the organizations they have already donated to and the success of those organizations was used to develop the model.

## Results 

### Data Preprocessing 

####  Target Variables
The following variable was the target that the model tried to predict: 
* IS_SUCCESSFUL - Describes whether the money was used effectively 

#### Feature Variables
The following variables were the features of the model that were used to help predict the target: 
* APPLICATION_TYPE - Type of application 
* AFFILIATION - Industry / Sector 
* CLASSIFICATION - Government organization classification 
* ORGANIZATION - Type of organization 
* STATUS - Whether the organization is still active
* INCOME_AMT - Amount of income 
* SPECIAL_CONSIDERATIONS - Special considerations to be taken into account for application 
* ASK_AMT - Amount of money requested 

#### Removed Variables 
The following variables and their data were removed because they were neither targets nor features: 
* EIN - ID 
* NAME - Name of the organization 
* USE_CASE - Reason for funding 

### Compiling, Training, and Evaluating the Model

#### Neurons, Layers & Activation Functions 
In the original model, I used the following: 
* Number of Neurons: 10 (1st layer - 7 & 2nd layer - 3)
* Number of Layers: 3 (2 hidden & 1 output)
* Activation Function: ELU, Sigmoid 

For the outer layer, I used the Sigmoid activation function because this is a binary classification problem. For the inner layer, originally the ReLU activation function was used because it's the most commonly used activation function and can handle non-linear relationships. The ELU activation function, which is a variant of the ReLU activation function, performed even better than the ReLU activation function so I selected ELU as my activacation function for the hidden layers. 

I selected a number of neurons that fell between 1 and the number of input features; after testing different variations of layers and number of neurons, I settled on using 2 hidden layers, 1 output layer and 10 neurons in total. 

#### Model Performance 
* The optimized version of the model was accurate 72.9% of the time. 

#### Model Optimization 
I performed the following steps to try and improve the performance (accuracy score) of the model: 
* Removed additional variable(s) from the features by testing the impact of their removal on the performance of the model.
* Decreased the number of epochs to reduce chances of overfitting; I selected the number of epochs by using the rule-of-thumb of 3 times the number of variables <a href="https://gretel.ai/gretel-synthetics-faqs/how-many-epochs-should-i-train-my-model-with#:~:text=The%20right%20number%20of%20epochs,again%20with%20a%20higher%20value">(Source)</a> and tested other amounts before settling on a final number of epochs. 
* Tested other activation functions and selected the best performing one. 
* Tested different combinations of hidden layers and number of neurons. <a href="https://www.linkedin.com/pulse/choosing-number-hidden-layers-neurons-neural-networks-sachdev">(Source)</a>

## Summary 
The initial model had an accuracy score of 69.7%, and the optimized model currently has an accuracy score of 72.9%. 

The ELU activation function provided better results. Having more hidden layers and more neurons in total didn't necessarily provide better results. 

Logistic regression could also be used instead of a deep learning neural network model to solve this problem as it is binary classification problem (successful or not successful), and we're looking to produce a model that predicts and quantifies the risk of donating to a non-profit organization.


