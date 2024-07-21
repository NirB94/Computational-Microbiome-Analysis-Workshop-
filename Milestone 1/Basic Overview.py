import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from skbio.stats.ordination import pcoa
import seaborn as sns

train_metadata = pd.read_csv(r"train_metadata.csv")
train_data = pd.read_csv(r"train_data.csv")

# Load the provided data
file_path = r'train_metadata.csv'
data = pd.read_csv(file_path)

# Convert collection_date to datetime
data['collection_date'] = pd.to_datetime(data['collection_date'])

# Count the number of samples for each baboon
sample_counts = data.groupby('baboon_id').size().reset_index(name='sample_count')

# Calculate time differences between samples for each baboon
data = data.sort_values(by=['baboon_id', 'collection_date'])
data['time_diff'] = data.groupby('baboon_id')['collection_date'].diff().dt.days

# Visualize the results

# Bar chart for number of samples per baboon
plt.figure(figsize=(24, 12))
sns.barplot(x='baboon_id', y='sample_count', data=sample_counts, palette='tab20')
plt.title('Number of Samples per Baboon')
plt.xlabel('Baboon ID')
plt.ylabel('Number of Samples')
plt.xticks(rotation=90)
plt.savefig('Number of Samples per Baboon.png')

# Scatter plot for time differences between samples
plt.figure(figsize=(14, 14))
sns.scatterplot(x='collection_date', y='baboon_id', hue='social_group', data=data, palette='tab20', s=50)
plt.title('Time Differences Between Samples for Each Baboon')
plt.xlabel('Collection Date')
plt.ylabel('Baboon ID')
plt.xticks(rotation=45)
# Set x-axis ticks to include all years from 2000 to 2014
plt.gca().set_xticks(pd.date_range(start='2000-01-01', end='2014-12-31', freq='YS'))
plt.gca().set_xticklabels(pd.date_range(start='2000-01-01', end='2014-12-31', freq='YS').year)
plt.legend(title='Social Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('Time Differences Between Samples for Each Baboon.png')

# Count the number of samples for each social group
group_counts = data.groupby('social_group').size().reset_index(name='sample_count')

# Bar chart for number of samples per social group
plt.figure(figsize=(12, 6))
sns.barplot(x='social_group', y='sample_count', data=group_counts, palette='tab20')
plt.title('Number of Samples per Social Group')
plt.xlabel('Social Group')
plt.ylabel('Number of Samples')
plt.xticks(rotation=90)
plt.savefig('Number of Samples per Social Group.png')

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
plt.savefig('PCoA of Bray-Curtis Distances')

# Calculate average distance between subsequent samples
# Assuming rows are sorted by time, we calculate the mean of the diagonal elements just above the main diagonal
num_samples = distance_matrix.shape[0]
subsequent_distances = [distance_matrix[i, i+1] for i in range(num_samples-1)]
average_distance = sum(subsequent_distances) / len(subsequent_distances)

print("Average distance between subsequent samples:", average_distance)
