# Set environment name
$env_name = "bda2"

# Activate the new environment
mamba create -n $env_name python=3.10
mamba activate $env_name
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to activate the environment $env_name"
    pause
    exit $LASTEXITCODE
}

# Install required packages
pip install astropy
pip install matplotlib
pip install nltk
pip install scikit-learn
pip install wordcloud
pip install scipy
pip install spacy
pip install gensim
pip install notebook
pip install ipykernel
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install required packages"
    pause
    exit $LASTEXITCODE
}

# Download NLTK data
python -m nltk.downloader stopwords punkt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to download NLTK data"
    pause
    exit $LASTEXITCODE
}

# Install Word2Vec and FastText
mamba install -n $env_name gensim
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install Word2Vec and FastText"
    pause
    exit $LASTEXITCODE
}

# Install the required packages for spaCy
mamba install -n $env_name spacy
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install spaCy"
    pause
    exit $LASTEXITCODE
}

# Download spaCy English model
python -m spacy download en
python -m spacy download en_core_web_sm
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to download spaCy English model"
    pause
    exit $LASTEXITCODE
}

# Verify the installation by importing the packages
python -c "import ast, string, nltk, matplotlib.pyplot as plt, spacy; from nltk.corpus import stopwords; from nltk.stem import PorterStemmer, WordNetLemmatizer; from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS, CountVectorizer; from sklearn.cluster import AgglomerativeClustering, KMeans; from sklearn.metrics import pairwise_distances, jaccard_score; from sklearn.metrics.pairwise import cosine_similarity; from wordcloud import WordCloud; from scipy.spatial.distance import pdist; from sklearn.decomposition import PCA, LatentDirichletAllocation, NMF; from scipy.cluster.hierarchy import linkage, dendrogram, fcluster; from gensim import corpora, models; from nltk.tokenize import word_tokenize; from nltk.util import ngrams; from collections import Counter; from gensim.models import Word2Vec, FastText"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to import packages in Python"
    pause
    exit $LASTEXITCODE
}

# Deactivate the environment
conda deactivate
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to deactivate the environment $env_name"
    pause
    exit $LASTEXITCODE
}
