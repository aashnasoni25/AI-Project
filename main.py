import rdkit
from rdkit import DataStructs
import rdkit.Chem as Chem
import pubchempy as pcp
from padelpy import from_smiles
import pandas as pd

from rdkit.Chem.Fingerprints import FingerprintMols
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
import plotly.offline as offline

import seaborn as sns
import pandas as pd

smiles_data = pd.read_csv("/Users/aashnasoni/SWnet/data/GDSC/GDSC_data/GDSC_smiles.csv")
print(smiles_data)

#testing

#1: C[C@H]1C[C@@H]([C@@H]([C@H](/C=C(/[C@@H]([C@H](/C=C\C=C(\C(=O)NC2=CC(=O)C(=C(C1)C2=O)NCC=C)/C)OC)OC(=O)N)\C)C)O)OC
#1242: CCN(CC)CCCCNC1=NC2=NC(=C(C=C2C=N1)C3=CC(=CC(=C3)OC)OC)NC(=O)NC(C)(C)C
#179: CC(CS(=O)(=O)C1=CC=C(C=C1)F)(C(=O)NC2=CC(=C(C=C2)C#N)C(F)(F)F)O

m1 = Chem.MolFromSmiles('C[C@H]1C[C@@H]([C@@H]([C@H](/C=C(/[C@@H]([C@H](/C=C\C=C(\C(=O)NC2=CC(=O)C(=C(C1)C2=O)NCC=C)/C)OC)OC(=O)N)\C)C)O)OC')
m2 = Chem.MolFromSmiles('CC(CS(=O)(=O)C1=CC=C(C=C1)F)(C(=O)NC2=CC(=C(C=C2)C#N)C(F)(F)F)O')

s1 = Chem.RDKFingerprint(m1)
s2 = Chem.RDKFingerprint(m2)

#s1 = FingerprintMols.FingerprintMol(m1)
#s2 = FingerprintMols.FingerprintMol(m2)

print(DataStructs.FingerprintSimilarity(s1, s2))

#1: with FingerprintMols.FingerprintMol - 0.3082758620689655
#2: with Chem.RDKFingerprint - 0.3082758620689655

numDrugs = smiles_data.shape[0]

similarity_data = np.zeros((numDrugs, numDrugs))

for i in range(numDrugs):
  for j in range(numDrugs):
    m1 = Chem.MolFromSmiles(smiles_data.iat[i,1])
    m2 = Chem.MolFromSmiles(smiles_data.iat[j,1])

    s1 = Chem.RDKFingerprint(m1)
    s2 = Chem.RDKFingerprint(m2)
    similarity_data[i,j] = DataStructs.FingerprintSimilarity(s1, s2)

similarity_df = pd.DataFrame(similarity_data)
print(similarity_df)

cell_drug_labels = pd.read_csv("/Users/aashnasoni/SWnet/data/GDSC/GDSC_data/cell_drug_labels.csv")
print(cell_drug_labels)

from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(10,7))
ax.hist(cell_drug_labels['labels'])

#fig, ax = plt.subplots(figsize=(10,50))
#plt.scatter(cell_drug_labels['cell_line_id'], cell_drug_labels['labels'])
#plt.show()

cell_drug_labels.groupby(['cell_line_id']).mean().reset_index().plot(kind='scatter',x='cell_line_id',y='labels')

cell_drug_labels.groupby(['drug_id']).mean().reset_index().plot(kind='scatter',x='drug_id',y='labels')


#Plot correlation heatmap
import seaborn as sns
gene_expression = pd.read_csv("/Users/aashnasoni/SWnet/data/GDSC/GDSC_data/GDSC_rma.csv")
import csv


#sns.heatmap(gene_expression.corr(), vmin=-1, vmax=1, annot=True)

data = []
sample_names = []

first = True

df = pd.read_csv("/Users/aashnasoni/SWnet/data/GDSC/GDSC_data/GDSC_rma.csv")
print(len(df.columns))

with open("/Users/aashnasoni/SWnet/data/GDSC/GDSC_data/GDSC_rma.csv") as csvfile:
  csv_reader = csv.reader(csvfile, delimiter = ",")
  for row in csv_reader:
    if first:
      genes = row[1:]
      first = False
    else:
      sample_names.append(row[0])
      data.append(row[1:])

data = np.array(data).astype(float)

print(data)
print(len(genes))
print(len(sample_names))
print(data.shape)

sns.set_context("paper", font_scale=0.3)
sns_plot = sns.clustermap(data, xticklabels=sample_names, yticklabels = genes)
sns_plot.savefig("heatmap.pdf")
plt.show()

#sns.heatmap(cell_drug_labels.corr(), vmin=-1, vmax=1, annot=True)
cell_drug_labels = pd.read_csv("/Users/aashnasoni/SWnet/data/GDSC/GDSC_data/cell_drug_labels.csv")
sns.set_context("paper", font_scale=0.3)
sns_plot = sns.clustermap(cell_drug_labels, xticklabels=cell_drug_labels["cell_line_id"], yticklabels = cell_drug_labels["drug_id"])
sns_plot.savefig("corr_heatmap.pdf")
plt.show()

scaler = StandardScaler()
X_train = gene_expression.apply(pd.to_numeric)
X_train_scl = scaler.fit_transform(X_train)
print(X_train[:3])

components = 21
pca = PCA(n_components=components)
Y = pca.fit(X_train_scl)
var_exp = Y.explained_variance_ratio_
cum_var_exp = np.cumsum(var_exp)

x = ["PC%s" %i for i in range(1,components)]
trace1 = go.Bar(
    x=x,
    y=list(var_exp),
    name="Explained Variance")

trace2 = go.Scatter(
    x=x,
    y=cum_var_exp,
    name="Cumulative Variance")

layout = go.Layout(
    title='Explained variance',
    xaxis=dict(title='Principle Components', tickmode='linear'))

data = [trace1, trace2]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig)