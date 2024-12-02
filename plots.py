from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Load your data
data = pd.read_csv("protein-kinase-pair/duolin-test/combined_with_label.csv")

# Label encode categorical features for embeddings
encoder = LabelEncoder()
data['kinase_encoded'] = encoder.fit_transform(data['kinase'])

# Concatenate sequence lengths and encoded kinase values as basic embeddings
data['sequence_length'] = data['Sequence'].apply(len)
data['kinase_sequence_length'] = data['kinase_sequence'].apply(len)

# Create feature matrix for PCA
embedding_features = data[['sequence_length', 'kinase_sequence_length', 'kinase_encoded']]

# Apply PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(embedding_features)
data['pca_1'] = pca_results[:, 0]
data['pca_2'] = pca_results[:, 1]

# Plot PCA results
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='pca_1', y='pca_2', hue='label', palette='coolwarm', s=60)
plt.title('PCA Plot of Sequence Embeddings by Label')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Label')
plt.show()
embeddings = np.vstack((data['sequence_length'], data['kinase_sequence_length'], data['kinase_encoded'])).T

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(embeddings)

# Add t-SNE results to the data
data['tsne_1'] = tsne_results[:, 0]
data['tsne_2'] = tsne_results[:, 1]

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='tsne_1', y='tsne_2', hue='label', palette='coolwarm', s=60)
plt.title('t-SNE Plot of Sequence Embeddings by Label')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='Label')
plt.show()
