# Step 1: Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from qiskit import QuantumCircuit, Aer, execute
from qiskit.utils import QuantumInstance

# Step 2: Load & preprocess dataset
iris = load_iris()
X = iris.data[:, :2]   # take first 2 features for clarity
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Define quantum feature embedding (angle encoding)
def quantum_feature_map(x):
    qc = QuantumCircuit(len(x))
    for i, val in enumerate(x):
        qc.ry(val, i)  # encode each feature as a rotation
    return qc

# Step 4: Encode data into statevectors
backend = Aer.get_backend('statevector_simulator')

def encode_dataset(X):
    states = []
    for x in X:
        qc = quantum_feature_map(x)
        result = execute(qc, backend).result()
        state = result.get_statevector(qc)
        states.append(np.real(state))  # use real part only for ML
    return np.array(states)

X_train_q = encode_dataset(X_train)
X_test_q  = encode_dataset(X_test)

# Step 5: Classical baseline with PCA
pca = PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Step 6: Train & evaluate classifiers
clf_q = SVC(kernel='linear').fit(X_train_q, y_train)
clf_pca = SVC(kernel='linear').fit(X_train_pca, y_train)

y_pred_q = clf_q.predict(X_test_q)
y_pred_pca = clf_pca.predict(X_test_pca)

print("Quantum embedding accuracy:", accuracy_score(y_test, y_pred_q))
print("PCA baseline accuracy:", accuracy_score(y_test, y_pred_pca))

# Step 7: Visualize embeddings with PCA for 2D view
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.scatter(X_train_pca[:,0], X_train_pca[:,1], c=y_train)
plt.title("PCA Embedding")

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
X_train_q_tsne = tsne.fit_transform(X_train_q)

plt.subplot(1,2,2)
plt.scatter(X_train_q_tsne[:,0], X_train_q_tsne[:,1], c=y_train)
plt.title("Quantum Embedding (t-SNE)")
plt.show()