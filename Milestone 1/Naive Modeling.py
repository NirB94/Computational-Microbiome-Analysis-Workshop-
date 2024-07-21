import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.spatial.distance import braycurtis
from sklearn.linear_model import LinearRegression

# Load microbiome and FFQ data
metadata_path = 'train_metadata.csv'
train_data_path = 'train_data.csv'

metadata = pd.read_csv(metadata_path)
traindata = pd.read_csv(train_data_path, index_col=0)

# Extract FFQ data (assuming FFQ data is in columns diet_PC1 to diet_PC13)
ffq_columns = [col for col in metadata.columns if col.startswith('diet_PC')]
ffq_data = metadata[ffq_columns]

# Merge microbiome data with FFQ data based on sample identifier
data = traindata.merge(metadata.set_index('sample'), left_index=True, right_index=True)

# Select only the numeric columns for microbiome data
microbiome_columns = traindata.columns
data_numeric = data[microbiome_columns]

# Fill NaN values with zero or another appropriate value
data_numeric = data_numeric.fillna(0)


# Define a naive prediction function
def naive_prediction(previous_data):
    return previous_data


# Define an enhanced prediction function using linear regression
class EnhancedPredictionModel:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, previous_data, ffq_data):
        combined_data = np.hstack([previous_data, ffq_data])
        return self.model.predict(combined_data.reshape(1, -1)).flatten()


# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
train_data_ffq, test_data_ffq = train_test_split(ffq_data, test_size=0.2, shuffle=False)
train_data_ffq = train_data_ffq.fillna(0)
test_data_ffq = test_data_ffq.fillna(0)

# Train the enhanced prediction model
enhanced_model = EnhancedPredictionModel()
X_train = np.hstack([train_data[microbiome_columns].values[:-1], train_data_ffq.values[:-1]])
y_train = train_data[microbiome_columns].values[1:]
enhanced_model.fit(X_train, y_train)

# Predict using the naive model
naive_predictions = []
for i in range(1, len(test_data)):
    naive_pred = naive_prediction(test_data.iloc[i - 1][microbiome_columns])
    naive_predictions.append(naive_pred)

# Calculate Bray-Curtis dissimilarity for naive model
naive_dissimilarities = []
for i in range(1, len(test_data)):
    dissimilarity = braycurtis(test_data.iloc[i][microbiome_columns], naive_predictions[i - 1])
    naive_dissimilarities.append(dissimilarity)

# Calculate average dissimilarity for the naive model
average_naive_dissimilarity = np.mean(naive_dissimilarities)

print("Average Bray-Curtis dissimilarity for naive model:", average_naive_dissimilarity)

# Predict using the enhanced model
enhanced_predictions = []
for i in range(1, len(test_data)):
    enhanced_pred = enhanced_model.predict(test_data.iloc[i - 1][microbiome_columns].values,
                                           test_data_ffq.iloc[i - 1].values)
    enhanced_predictions.append(enhanced_pred)

# Calculate Bray-Curtis dissimilarity for enhanced model
enhanced_dissimilarities = []
for i in range(1, len(test_data)):
    dissimilarity = braycurtis(test_data.iloc[i][microbiome_columns], enhanced_predictions[i - 1])
    enhanced_dissimilarities.append(dissimilarity)

# Calculate average dissimilarity for the enhanced model
average_enhanced_dissimilarity = np.mean(enhanced_dissimilarities)

print("Average Bray-Curtis dissimilarity for enhanced model:", average_enhanced_dissimilarity)
