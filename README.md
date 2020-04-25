# Modeling Graph Convolutional Networks to Predict Compounds' Solubilities Using DeepChem

This study focused on utilizing the DeepChem package to effectively predict the solubilities of chemical compounds. Solubility is the measure of how fast a compound can dissolve in solution. This is important for drug-discovery and applied chemistry techniques.

Using a sample CSV data from DeepChem repository, the parameters were selected directly from the data to explore. After defining the parameters for the model, a transformer was initiated in order to split the data and load features.

Pearson's r^2 coefficient is used to measure accuracy, number of features was set to 75, while the batch size is 128. 

In this study Graph Convolutional Networks (GCN) were used because they provide easy manipulation to simulate chemical compounds. Following Stephen Wolfram's newly published article "Finally We May Have a Path to the Fundamental Laws of Physics... and It's Beautiful" (https://writings.stephenwolfram.com/2020/04/finally-we-may-have-a-path-to-the-fundamental-theory-of-physics-and-its-beautiful/), he suggests atoms being nodes and bonds being edges in a hypergraph. This metaphor is perfect for GCNs because it could simulate entire compounds from convoluted layers of hypergraphs. GCNs have been proven very effective to train models, however they have the huge limitation of only working with molecular graphs.

After deciding on the model, the data is fit to be trained and tested. Metrics are computed predicting a 95% accuracy rate.

Finally, after designing the model, a simple example was tested to predict the solubility of random molecules using SMILES notation.

This study follows the original paper from Stanford/Schrodinger Inc's MoleculeNet: A Benchmark for Molecular Machine Learning. (https://arxiv.org/pdf/1703.00564.pdf)
