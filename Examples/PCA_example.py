from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

data = load_breast_cancer()

X, y = data['data'], data['target']

U, S, V_t = np.linalg.svd(X-np.mean(X, axis=0))

colors = ['red','green']
i, j = 4, 29

plot = plt.scatter(X[:,i], X[:,j], c=y, alpha=0.35, cmap=matplotlib.colors.ListedColormap(colors))
plt.legend(handles=plot.legend_elements()[0], labels=list(data['target_names']))
plt.xlabel(data['feature_names'][i])
plt.ylabel(data['feature_names'][j])
plt.title('Two attributes of breast cancer data')
plt.show()

i, j = 3, 20

plot = plt.scatter(X[:,i], X[:,j], c=y, alpha=0.35, cmap=matplotlib.colors.ListedColormap(colors))
plt.legend(handles=plot.legend_elements()[0], labels=list(data['target_names']))
plt.xlabel(data['feature_names'][i])
plt.ylabel(data['feature_names'][j])
plt.title('Two more indicative attributes of breast cancer data')
plt.show()

pca = PCA(n_components=3)
Xt = pca.fit_transform(X)

print('How much of the data gets explained by each of the \nfirst three principal components?')
print(pca.explained_variance_ratio_)
print('New plot point for first sample after PCA:')
print(Xt[0,0], Xt[0,1])
print('Same result using SVD (adjusted for sign):')
print(np.inner(-(X-np.mean(X, axis=0))[0], V_t[0]),\
       np.inner((X-np.mean(X, axis=0))[0], V_t[1]))
plot = plt.scatter(Xt[:,0], Xt[:,1], c=y, alpha=0.35, cmap=matplotlib.colors.ListedColormap(colors))
plt.legend(handles=plot.legend_elements()[0], labels=list(data['target_names']))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Breast cancer data projected on first two PCs')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Xt[:,0], Xt[:,1], Xt[:,2], c=y, alpha=0.35, cmap=matplotlib.colors.ListedColormap(colors))
fig.legend(handles=plot.legend_elements()[0], labels=list(data['target_names']))
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.title('Breast cancer data projected on first three PCs')
plt.show()
