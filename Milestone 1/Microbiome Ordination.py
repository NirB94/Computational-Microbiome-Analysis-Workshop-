import pandas as pd
from scipy.spatial.distance import pdist, squareform
from skbio.stats.ordination import pcoa
import matplotlib.pyplot as plt
import numpy as np

# Load the microbiome data
file_path = r"train_data.csv"  # Update this with the correct path to your CSV file
train_data = pd.read_csv(file_path, index_col=0)

# Load metadata
metadata_path = 'train_metadata.csv'  # Update this with the correct path to your metadata file
metadata = pd.read_csv(metadata_path, index_col=0)

# Ensure only numeric data is used for Bray-Curtis calculation
numeric_data = train_data.select_dtypes(include=[float, int])

# Calculate Bray-Curtis distances
bray_curtis_distances = pdist(numeric_data.values, metric='braycurtis')
distance_matrix = squareform(bray_curtis_distances)

# Perform PCoA
pcoa_results = pcoa(distance_matrix)

# Plot PCoA results
plt.figure(figsize=(12, 8))
plt.scatter(pcoa_results.samples['PC1'], pcoa_results.samples['PC2'], alpha=0.5)
plt.title('PCoA of Bray-Curtis Distances')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.savefig('PCoA of Bray-Curtis Distances.png')

# Calculate average distance between subsequent samples
# Assuming rows are sorted by time, we calculate the mean of the diagonal elements just above the main diagonal
num_samples = distance_matrix.shape[0]
subsequent_distances = [distance_matrix[i, i + 1] for i in range(num_samples - 1)]
average_distance = sum(subsequent_distances) / len(subsequent_distances)

print("Average distance between subsequent samples:", average_distance)

# Ensure only numeric data is used for analysis
numeric_data = train_data.select_dtypes(include=[float, int])

# Merge the train_data and metadata on the sample identifier
# Assuming 'sample' is the common identifier for merging
merged_df = numeric_data.join(metadata.reset_index().set_index('sample'), how='inner')


# Function to calculate autocorrelation
def autocorrelation(x, lag):
    return np.corrcoef(x[:-lag], x[lag:])[0, 1]


# Calculate Bray-Curtis distances between samples for each baboon
autocorr_results = {}
for subject in merged_df['baboon_id'].unique():
    subject_data = merged_df[merged_df['baboon_id'] == subject].sort_values(by='collection_date')
    subject_numeric_data = subject_data[numeric_data.columns]
    bray_curtis_distances = pdist(subject_numeric_data.values, metric='braycurtis')
    distance_matrix = squareform(bray_curtis_distances)
    # Take the distances between subsequent samples
    subsequent_distances = [distance_matrix[i, i + 1] for i in range(distance_matrix.shape[0] - 1)]
    if len(subsequent_distances) > 1:  # Ensure there's enough data to compute autocorrelation
        autocorr = [autocorrelation(subsequent_distances, lag) for lag in
                    range(1, min(10, len(subsequent_distances)))]  # Calculate up to lag 10
        autocorr_results[subject] = autocorr


# Visualize autocorrelation results
def plot_autocorrelation(autocorr_results, subject):
    lags = range(1, len(autocorr_results[subject]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(lags, autocorr_results[subject], marker='o', linestyle='-')
    plt.title(f'Autocorrelation of Bray-Curtis Distances for {subject}')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    plt.savefig(f'Autocorrelation of Bray-Curtis Distances for {subject}.png')


# Example usage: Visualize the autocorrelation for a specific subject
subject_example = 'Baboon_4'  # Replace with a valid subject ID from your data

if subject_example in autocorr_results:
    plot_autocorrelation(autocorr_results, subject_example)
else:
    print(f"No autocorrelation data for {subject_example}")


# Function to calculate autocorrelation
def autocorrelation(x, lag):
    return np.corrcoef(x[:-lag], x[lag:])[0, 1]


# Calculate mean of features across baboons in each social group and their subsequent Bray-Curtis distances
autocorr_results = {}
for social_group in merged_df['social_group'].unique():
    group_data = merged_df[merged_df['social_group'] == social_group]
    # Calculate mean of features across baboons in the same social group for each collection date
    mean_data = group_data.groupby('collection_date')[numeric_data.columns].mean()
    bray_curtis_distances = pdist(mean_data.values, metric='braycurtis')
    distance_matrix = squareform(bray_curtis_distances)
    # Take the distances between subsequent samples
    subsequent_distances = [distance_matrix[i, i + 1] for i in range(distance_matrix.shape[0] - 1)]
    if len(subsequent_distances) > 1:  # Ensure there's enough data to compute autocorrelation
        autocorr = [autocorrelation(subsequent_distances, lag) for lag in
                    range(1, min(10, len(subsequent_distances)))]  # Calculate up to lag 10
        autocorr_results[social_group] = autocorr

# Visualize autocorrelation results for all social groups in the same graph
plt.figure(figsize=(12, 8))
for social_group, autocorr in autocorr_results.items():
    lags = range(1, len(autocorr) + 1)
    plt.plot(lags, autocorr, marker='o', linestyle='-', label=social_group)

plt.title('Autocorrelation of Bray-Curtis Distances for All Social Groups')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.legend(title='Social Group')
plt.savefig('Autocorrelation of Bray-Curtis Distances for All Social Groups.png')
