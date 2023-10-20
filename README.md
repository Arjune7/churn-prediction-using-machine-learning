# Churn Prediction Model Developement

## Problem Statement

The goal of this project is to predict customer churn based on historical data. Churn prediction helps businesses identify customers who are likely to leave, allowing for targeted retention strategies.

## Data Collection

The dataset used for this project was collected from customer_churn_dataset.csv present inside the repository.

## Data Preprocessing

### Data Cleaning

- In preparing our dataset for machine learning models, I employed a systematic approach to ensure data consistency, appropriate format, and anomaly-free entries. First, I opted to drop rows with missing values, maintaining data integrity while removing limited missing data points. This step is crucial to ensure the quality of the dataset.

Next, I tackled categorical variables like 'Gender' and 'Location' using label encoding. This transformation of categorical values into numerical representations is essential for machine learning algorithms, allowing them to interpret the data effectively. Label encoding was chosen due to its simplicity and suitability for variables with ordinal relationships.

To standardize numerical features ('Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB'), I utilized the StandardScaler. Standardization is vital as it brings all features to a similar scale, preventing certain features from dominating others during the modeling process. This step ensures fair treatment of all features, leading to more accurate model predictions.

Lastly, addressing outliers was pivotal. Outliers can significantly impact the performance of machine learning algorithms, particularly regression and clustering models. I used the Interquartile Range (IQR) method to identify and remove outliers. By establishing lower and upper bounds (Q1 - 1.5 * IQR and Q3 + 1.5 * IQR, respectively), I ensured extreme values did not adversely affect the learning process of the models, resulting in more reliable predictions.

By implementing these strategies, I aimed to enhance the dataset's quality, creating a robust foundation for the machine learning models. These meticulous steps were instrumental in producing accurate and meaningful predictions regarding customer churn.

### Feature Selection
- In the quest for more nuanced insights, I engineered additional features that could potentially reveal deeper patterns in our dataset. Subscription to Age Ratio was introduced, aiming to capture the relationship between the length of subscription (in months) and the customer's age. This metric might unveil behavioral patterns based on varying subscription durations concerning age demographics.

Additionally, Monthly Bill per GB (Cost Per GB) was created to assess the cost efficiency of data usage. This feature calculates how much a customer pays for each gigabyte of data used, shedding light on the economic aspect of their usage behavior.

To ensure consistency and meaningful comparison across all features, I applied standard scaling to these newly engineered columns: 'Subscription_to_Age_Ratio' and 'Cost_Per_GB'. By employing the StandardScaler, I transformed these features to a similar scale, preventing any potential bias that might arise due to differing units or scales. This standardization process was vital, ensuring all features were treated equally and facilitating more accurate analyses and model predictions.

### Data Splitting
- To facilitate robust model training and evaluation, I split the dataset into training and testing subsets using the train_test_split function from the scikit-learn library. With a predetermined test size of 20% and a random state set to 42, I ensured the reproducibility of the split, maintaining consistency across multiple runs of the code. This division into training and testing sets allows for thorough training on one subset and unbiased evaluation on another, ensuring the model's ability to generalize well to unseen data.

## Model Selection

### Logistic Regression
- **Reason for selection:** I opted for Logistic Regression as my initial choice for modeling due to its simplicity, interpretability, and efficiency, especially when dealing with binary classification tasks like churn prediction. Logistic Regression is a well-established algorithm that forms a solid baseline for comparison with more complex models. Its linear nature makes it computationally efficient, allowing for quick training and prediction on large datasets, which is particularly beneficial when working with extensive customer churn data.

Additionally, Logistic Regression provides easily interpretable results, making it easier to communicate findings to stakeholders and understand the impact of different features on the prediction outcome. By starting with Logistic Regression, I could quickly assess the basic performance of the model and establish a foundation upon which more complex algorithms could be built and evaluated.

### Random Forest
- **Reason for selection:** Upon observing the initial results and understanding the complexity of the customer churn prediction task, I opted for Random Forest as my subsequent model choice. The decision to transition to Random Forest was based on its strengths in ensemble learning. By aggregating predictions from multiple decision trees, Random Forest mitigates overfitting issues, ensuring robust performance on diverse datasets like customer churn data.

Moreover, Random Forest's ability to provide insights into feature importance was pivotal. Identifying key factors influencing churn is crucial for devising targeted business strategies. Random Forest's capability to capture non-linear relationships and interactions within the data was also instrumental. In customer churn analysis, where customer behavior is influenced by a multitude of factors in complex ways, Random Forest's flexibility proved invaluable.

Lastly, Random Forest's robustness to outliers and noisy data points instilled confidence in the model's reliability. This was especially relevant in real-world scenarios where data imperfections are common. Overall, Random Forest's adaptability, interpretability, and robustness made it a compelling choice for enhancing the accuracy and depth of my churn prediction efforts.

### Gradient Boosting Machine
- **Reason for selection:** I chose the Gradient Boosting Machine (GBM) model as my primary tool for customer churn prediction due to its exceptional ability to handle intricate patterns within the data. GBM's iterative learning approach, where it continuously improves by focusing on misclassified data points, ensures high accuracy. Its flexibility in handling diverse data types, robustness against outliers, and capacity to capture complex relationships made it an ideal choice for the nuanced task of predicting customer behavior. In summary, GBM's adaptability, predictive power, and capability to handle real-world data challenges made it my top choice for this project.

## Model Evaluation

-In the initial evaluation phase before model optimization, I chose to employ the ROC curve alongside precision, recall, and F1-score metrics to comprehensively assess the performance of various machine learning models, including Logistic Regression and Random Forest, in predicting customer churn. The ROC curve, with its corresponding Area Under the Curve (AUC) score, provided a holistic view of the trade-off between true positive rate and false positive rate. A higher AUC value signifies a more effective model in distinguishing between churn and non-churn instances.

Additionally, I focused on precision, recall, and F1-score to delve deeper into the model's performance, particularly concerning its ability to minimize false positives and false negatives. Precision illuminated the accuracy of positive predictions, emphasizing the model's reliability in identifying actual churn cases. Recall, on the other hand, measured the model's capability to capture all positive instances, shedding light on its sensitivity to churn patterns. The F1-score, being the harmonic mean of precision and recall, provided a balanced assessment, considering both types of misclassifications.

This evaluation approach allowed for a nuanced understanding of each model's strengths and weaknesses, laying the groundwork for subsequent optimizations aimed at enhancing their predictive capabilities.

## Hyperparameter Optimization

- Employing hyperparameter tuning through grid search coupled with cross-validation was a strategic choice deeply rooted in the pursuit of model optimization and performance enhancement. Grid search, as a systematic method, allowed me to explore a range of hyperparameter combinations comprehensively. By specifying different values for key parameters, I could methodically test the model's sensitivity to these variations, ensuring no stone was left unturned in the search for the best configuration.

The synergy between grid search and cross-validation was pivotal. Cross-validation, in essence, divided my dataset into subsets, validating the model iteratively on different portions and training it on the remaining data. This iterative validation process offered a more realistic estimate of the model's performance, capturing its robustness and generalizability. By integrating grid search within this cross-validation framework, I not only optimized my models but did so in a manner that was unbiased and less prone to overfitting.

This approach was akin to fine-tuning a musical instrument. Each iteration of grid search within a cross-validation fold brought subtle adjustments, optimizing the model's parameters until it harmonized perfectly with the data. The amalgamation of these techniques, therefore, was not just a procedural choice; it was a strategic move to ensure my models were finely tuned, resilient, and capable of delivering reliable predictions in real-world scenarios.

## Model Deployment

- The trained Random Forest model was saved to a file using `joblib.dump()`.
- I chose the Random Forest algorithm for deployment due to its outstanding accuracy and robust performance observed during the model evaluation process. Its ability to handle complex relationships within the data and mitigate overfitting made it a reliable choice for making predictions on new customer data. To facilitate seamless deployment and interaction with the model, I opted for Flask, a lightweight and versatile web framework for Python. By leveraging Flask, I could create an API endpoint, allowing the model to receive input data and provide churn predictions efficiently. This combination of a powerful algorithm and a user-friendly deployment framework ensures that the model is not only accurate but also readily accessible for real-time predictions in production environments.


## Extra information
- For an in-depth exploration of the model development process, visualizations, and metrics, please refer to the Jupyter Notebook provided in this repository. This was developed using google colabotory, kindly for any questions or inquiries, feel free to reach out. Happy predicting!
