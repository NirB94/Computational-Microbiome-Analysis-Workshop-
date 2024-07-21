import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from skbio.stats.ordination import pcoa

# Load metadata
metadata_path = 'train_metadata.csv'
metadata = pd.read_csv(metadata_path)

# Extract FFQ data (assuming FFQ data is in columns diet_PC1 to diet_PC13)
ffq_columns = [col for col in metadata.columns if col.startswith('diet_PC')]
ffq_data = metadata[ffq_columns]

# Calculate Bray-Curtis distances
bray_curtis_distances = pdist(ffq_data.values, metric='braycurtis')
distance_matrix = squareform(bray_curtis_distances)

# Perform PCoA
pcoa_results = pcoa(distance_matrix)

# Add PCoA results to the metadata DataFrame
metadata.set_index(pcoa_results.samples.index, inplace=True)
metadata['PC1'] = pcoa_results.samples['PC1']
metadata['PC2'] = pcoa_results.samples['PC2']


# Plot the first two principal coordinates
def plot_pcoa(metadata, color_by):
    plt.figure(figsize=(12, 8))
    for group in metadata[color_by].unique():
        group_data = metadata[metadata[color_by] == group]
        plt.scatter(group_data['PC1'], group_data['PC2'], label=group, alpha=0.6)
    plt.title(f'PCoA of FFQ Data Colored by {color_by}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title=color_by)
    plt.grid(True)
    plt.savefig(f'PCoA of FFQ Data Colored by {color_by}.png')


# Plot PCoA colored by social group
plot_pcoa(metadata, 'social_group')

# Plot PCoA colored by season
plot_pcoa(metadata, 'season')
