# Knowledge Graph Embeddings using Textual Associations
This repository contains code and data for the experiments in the paper:

[Unsupervised Embedding Enhancements of Knowledge Graphs using Textual Associations](http://www.eecg.utoronto.ca/~veneris/ijcai19.pdf)\
by Neil Veira, Brian Keng, Kanchana Padmanabhan, Andreas Veneris\
Published at IJCAI 2019.

### Requirements
The code requires the following packages: 
 - Python 3.6
 - numpy 1.15.4
 - scipy 1.0.0
 - scikit-learn 0.19.1
 - gensim 3.5
 - nltk 3.3
 - tensorflow 1.5

### Usage
All experimental configurations can be run through the script run.py (see run.py --help for details), which performs the data processing, training, and evaluation steps. Seven bash scripts are provided to reproduce the seven configurations from the paper. The command format is ```run_<model>.sh <dataset> <scoring function> ```.

Supported datasets include Wordnet (``WN``) and Freebase (``FB``)

Supported scoring functions include Structured Embeddings (``SE``), Translational Embeddings (``TransE``), Translational Relations (``TransR``), RESCAL (``RESCAL``), DistMult (``DistMult``), and Holographic Embeddings (``HolE``). 

For example, to run the FeatureSum model on Wordnet using the DistMult scoring function, run

```
./run_FeatureSum.sh WN DistMult 
```
