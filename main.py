import ast
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import nltk
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import pairwise_distances, jaccard_score
import ast
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from scipy.spatial.distance import pdist
from nltk.tokenize import word_tokenize
from collections import defaultdict  

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

data = {}
titles = []

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

additionalStopWordsPreprocess = ['']
stopWordsPreprocess = stopwords.words('english') + additionalStopWordsPreprocess

filePaths = ["database_systems_publications.txt", "data_mining_publications.txt"]
filePath = filePaths[1]

amount_clusters_per_interval = [5, 10, 15, 5, 5, 5, 5]
index_timedrift = 0
time_interval = 10
time_overlap = 2
NGRAMS = (1,3)

def preProcessTitle(title, preProcessWay):
    '''
        Preprocess title in given way.
    '''
    title = title.lower()
    title = title.translate(str.maketrans('', '', string.punctuation))
    title = ' '.join(word for word in title.split() if word not in stopWordsPreprocess)

    if preProcessWay == 'lemmatization':
        title = ' '.join(lemmatizer.lemmatize(word) for word in title.split())
        return title
    elif preProcessWay == 'stemming':
        title = ' '.join(stemmer.stem(word) for word in title.split())
        return title
    else:
        return title

def preProcessData(preProcessWay):
    '''
        Preprocess data in a given way.
        Param: preProcessWay ["", "stemming", "lemmatization"]
    '''
    for year, titles_per_year in data.items():
        preProcessedTitles = [preProcessTitle(title, preProcessWay) for title in titles_per_year]
        data[year] = preProcessedTitles
        titles.extend(preProcessedTitles)

def readData():
    '''
        Reading data from text file.
    '''
    with open(filePath, 'r', encoding='utf-8') as file:
        for line in file:
            data_entry = ast.literal_eval(line)

            year = data_entry.get('year', '')
            title = data_entry.get('title', '')

            if year not in data:
                data[year] = []
            data[year].append(title)
    
def analyze_clusters(clustered_titles):
    # Iterate through each cluster
    for cluster_id, titles_in_cluster in clustered_titles.items():
        # Combine titles in the cluster into a single string
        cluster_text = ' '.join(titles_in_cluster)

        # Tokenize the text into words
        words = word_tokenize(cluster_text)

        # Remove stop words
        stop_words = set(stopwords.words('english'))
        filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

        # Create trigrams
        trigrams = list(nltk.ngrams(filtered_words, 3))

        # Calculate the frequency of each trigram
        trigram_freq = nltk.FreqDist(trigrams)

        # Print the most occurring trigrams in the cluster
        MOST_N_TOPICS = 20
        print(f"Cluster {cluster_id + 1} - Most Common Trigrams:")
        print(f"Count of titles in cluster: {len(titles_in_cluster)}")
        #print(titles_in_cluster)
        for trigram, count in trigram_freq.most_common(MOST_N_TOPICS):
            print(f"- {' '.join(trigram)}: {count}")

        # top_trigrams = dict(trigram_freq.most_common(MOST_N_TOPICS))
        # top_trigrams_str = {' '.join(trigram): count for trigram, count in top_trigrams.items()}

        # # Create a word cloud for the top trigrams
        # wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_trigrams_str)
        # plt.figure(figsize=(10, 6))
        # plt.imshow(wordcloud, interpolation='bilinear')
        # plt.axis('off')
        # plt.title(f'Top {MOST_N_TOPICS} Most Occurring Trigrams in Titles')
        # plt.show()

def main():
    readData()
    preProcessWays = ['', 'stemming', 'lemmatization']
    preProcessData(preProcessWays[2])

def analyze_whole_dataset():
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(titles)

    
    num_clusters = 20
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    kmeans.fit(tfidf_matrix)
    clusters = kmeans.labels_

    # Organize titles into clusters
    clustered_titles = defaultdict(list)

    print()
    for i, label in enumerate(clusters):
        clustered_titles[label].append(titles[i])

    # Analyze trigrams within each cluster
    analyze_clusters(clustered_titles)

def analyze_clusters(clustered_titles, interval_str):

    for cluster_id, titles_in_cluster in clustered_titles.items():
        cluster_text = ' '.join(titles_in_cluster)
        words = word_tokenize(cluster_text)

        # Create Ngrams
        ngrams = list(nltk.ngrams(words, NGRAMS))
        # Calculate freq of ngrams
        ngrams_freq = nltk.FreqDist(ngrams)

        print()
        print(f"Cluster {cluster_id + 1}:")
        print(f"Count of titles in cluster: {len(titles_in_cluster)}")
        print()
        for trigram, count in ngrams_freq.most_common(10):
            print(f"- {' '.join(trigram)}: {count}")


        # Create Wordcloud
        # top_ngrams_freq = dict(ngrams_freq.most_common(MOST_N_TOPICS))
        # top_ngrams_freq_str = {' '.join(trigram): count for trigram, count in top_ngrams_freq.items()}
        # wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_ngrams_freq_str)
        # plt.figure(figsize=(10, 6))
        # plt.imshow(wordcloud, interpolation='bilinear')
        # plt.axis('off')
        # plt.title(f'Cluster {cluster_id + 1}: Most occuring ngrams.')
        # plt.show()

        print()

def analyze_interval(titles_drift_interval, interval_str):
    vectorizer = TfidfVectorizer(ngram_range=NGRAMS)
    tfidf_matrix = vectorizer.fit_transform(titles_drift_interval)

    kmeans = KMeans(n_clusters=amount_clusters_per_interval[index_timedrift], random_state=42, init='k-means++', n_init=10)
    kmeans.fit(tfidf_matrix)

    clustered_titles = defaultdict(list)

    for i, label in enumerate(clusters):
        clustered_titles[label].append(titles[i])

    clustered_titles = dict(sorted(clustered_titles.items()))
    analyze_clusters(clustered_titles, interval_str)

def visualize_clusters_evolution(cluster_sizes, years, index_timedrift):
    plt.figure(figsize=(10, 6))
    plt.plot(years, cluster_sizes, marker='o')
    plt.title(f'Cluster Evolution Over Time (Interval {index_timedrift})')
    plt.xlabel('Year')
    plt.ylabel('Number of Publications in a Cluster')
    plt.grid(True)
    plt.show()

def visualize_word_cloud(cluster_center, cluster_count):
    print(cluster_center)
    plt.figure(figsize=(8, 8))
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          min_font_size=10).generate_from_frequencies(dict(cluster_center))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title(f'Word Cloud for Cluster {cluster_count}')
    plt.show()


def analyze_clustered_titles(titles_drift_interval, amount_clusters_per_interval, index_timedrift):
    vectorizer = TfidfVectorizer(ngram_range=NGRAMS)
    tfidf_matrix = vectorizer.fit_transform(titles_drift_interval)

    kmeans = MiniBatchKMeans(n_clusters=amount_clusters_per_interval[index_timedrift], random_state=42, init='k-means++', n_init=10)
    kmeans.fit(tfidf_matrix)

    clusters = kmeans.labels_
    cluster_len_titles = [0] * amount_clusters_per_interval[index_timedrift]

    feature_names = vectorizer.get_feature_names_out()

    for label in clusters:
        cluster_len_titles[label] += 1

    cluster_centers = [sorted(zip(center, feature_names), reverse=True) for center in kmeans.cluster_centers_]
    clusters = sorted(zip(cluster_len_titles, cluster_centers), key=lambda x: x[0], reverse=True)
        
    cluster_count = 0
    for  cluster_size, cluster_center in clusters:
        relative_cluster_size = cluster_size / len(titles_drift_interval)
        cluster_count+= 1
        features = [feature for feature in cluster_center if feature[0] >= 0.01]

        for j in range(len(features)):
            existing_feature = features[j]
            for feature in features[j + 1:]:
                if ' ' + existing_feature[1] in ' ' + feature[1] and existing_feature[0] == feature[0]:
                    break
                elif ' ' + feature[1] in ' ' + existing_feature[1] and existing_feature[0] == feature[0]:
                    features[j] = feature
                    break
            else:
                continue
            break
        else:
            print(f'Cluster {cluster_count}: {relative_cluster_size * 100:.2f}%')
            for feature in sorted(features, reverse=True):
                print(f'        {feature[0]:.2f} {feature[1]}')
    


    # years = list(range(1960 + index_timedrift * drift_interval, 1960 + (index_timedrift + 1) * drift_interval))
    # visualize_clusters_evolution(cluster_len_titles, years, index_timedrift)

    # # Select a smaller number of relevant clusters for detailed exploration
    # clusters_to_visualize = clusters[:2]  # Adjust the number as needed

    # for cluster_count, (cluster_size, cluster_center) in enumerate(clusters_to_visualize, start=1):
    #     relative_cluster_size = cluster_size / len(titles_drift_interval)
    #     print(f'Cluster {cluster_count}: {relative_cluster_size * 100:.2f}%')
    #     for feature in sorted(cluster_center, key=lambda x: x[0], reverse=True):
    #         print(f'        {feature[0]:.2f} {feature[1]}')

    #     # Word Cloud Visualization
    #     visualize_word_cloud(dict(cluster_center), cluster_count)


main()
print()
print("Topic Drift:")
print(f"Time interval: {time_interval} years.")
print(f"Time overlap: {time_overlap} years.")
print(f"Ngrams set to: {NGRAMS}.")
print()
for year in range(1960, 2025, time_interval):
    start_year = year - time_overlap
    titles_drift_interval = []

    for i in range(time_interval + time_overlap*2):
        if str(start_year + i) in data.keys():
            titles_drift_interval.extend(data[str(start_year + i)])

    if len(titles_drift_interval) == 0:
        index_timedrift+=1
        continue
    
    print('-------------------------------------------------------------------------------')
    interval_str = str(start_year) + '-'+str(start_year+i)
    print(f"Clustering {len(titles_drift_interval)} titles in range: {start_year}-{start_year+i}")

    analyze_clustered_titles(titles_drift_interval, amount_clusters_per_interval, index_timedrift)
    # print('-------------------------------------------------------------------------------')
    index_timedrift+=1