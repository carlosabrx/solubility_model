#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Predicting Compound Solubilities using DeepChem

Carlos Abreu 

April 2020
'''

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
import deepchem as dc
import numpy as np
import tensorflow as tf
from deepchem.models import GraphConvModel


# In[2]:


data = open('delaney-processed.csv')


# In[3]:


delaney_tasks = ['measured log solubility in mols per litre']
featurizer = dc.feat.ConvMolFeaturizer()
loader = dc.data.CSVLoader(tasks=delaney_tasks, smiles_field="smiles", featurizer=featurizer)
dataset = loader.featurize(data, shard_size=8192)

  # Initialize transformers
transformers = [dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset)]
print("About to transform data")

for transformer in transformers:
    dataset = transformer.transform(dataset)
    splitter = dc.splits.RandomSplitter()
    train, valid, test = splitter.train_valid_test_split(dataset)


# In[4]:


# Fit models
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

# Number of features 
n_feat = 75

# Batch size of models
batch_size = 128

model = GraphConvModel(len(delaney_tasks), batch_size=batch_size, mode='regression', dropout=0.2)


# In[5]:


# Fit trained model
model.fit(train, nb_epoch=100)

print("Evaluating model")
train_scores = model.evaluate(train, [metric], transformers)
valid_scores = model.evaluate(valid, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)


# In[6]:


#Predictability test for trained model

smiles = ['COC(C)(C)CCCC(C)CC=CC(C)=CC(=O)OC(C)C','CCOC(=O)CC','CSc1nc(NC(C)C)nc(NC(C)C)n1','CC(C#C)N(C)C(=O)Nc1ccc(Cl)cc1','Cc1cc2ccccc2cc1C']
mols = [Chem.MolFromSmiles(s) for s in smiles]
featurizer = dc.feat.ConvMolFeaturizer()
x = featurizer.featurize(mols)
predicted_solubility = model.predict_on_batch(x)
print(predicted_solubility)


#Compound Structures

mol = Chem.MolFromSmiles("COC(C)(C)CCCC(C)CC=CC(C)=CC(=O)OC(C)C")
mol1 = Chem.MolFromSmiles("CCOC(=O)CC")
mol2 = Chem.MolFromSmiles("CSc1nc(NC(C)C)nc(NC(C)C)n1")
mol3 = Chem.MolFromSmiles("CC(C#C)N(C)C(=O)Nc1ccc(Cl)cc1")
mol4 = Chem.MolFromSmiles("Cc1cc2ccccc2cc1C")

