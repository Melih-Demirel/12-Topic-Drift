#!/bin/bash

env_name="bda2"
# Create a new environment named 'my_env'
mamba create -n $env_name python=3.10

# Activate the new environment
mamba activate $env_name

# Install required packages
mamba install -n $env_name astropy matplotlib nltk scikit-learn wordcloud scipy spacy gensim

# Install additional packages not available in the conda repositories
pip install nltk wordcloud
pip install notebook
# Download NLTK data
python -m nltk.downloader stopwords punkt

# Install Word2Vec and FastText
mamba install -n $env_name gensim

# Install the required packages for spaCy
mamba install -n $env_name spacy

# Download spaCy English model
python -m spacy download en
python -m spacy download en_core_web_sm
# Verify the installation by importing the packages
python -c "import ast, string, nltk, matplotlib.pyplot as plt, spacy; from nltk.corpus import stopwords; from nltk.stem import PorterStemmer, WordNetLemmatizer; from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS, CountVectorizer; from sklearn.cluster import AgglomerativeClustering, KMeans; from sklearn.metrics import pairwise_distances, jaccard_score; from sklearn.metrics.pairwise import cosine_similarity; from wordcloud import WordCloud; from scipy.spatial.distance import pdist; from sklearn.decomposition import PCA, LatentDirichletAllocation, NMF; from scipy.cluster.hierarchy import linkage, dendrogram, fcluster; from gensim import corpora, models; from nltk.tokenize import word_tokenize; from nltk.util import ngrams; from collections import Counter; from gensim.models import Word2Vec, FastText"

# Deactivate the environment
conda deactivate
