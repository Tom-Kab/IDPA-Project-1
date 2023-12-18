import pandas as pd
import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from IPython.display import clear_output


indexName = "all_content"

try:
    es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("IDPA","project1"),
    )
except ConnectionError as e:
    print("Connection Error:", e)
    
if es.ping():
    print("Succesfully connected to ElasticSearch!!")
else:
    print("Oops!! Can not connect to Elasticsearch!")

def plot_agglomerative_dendrogram(data, linkage_method='complete'):
    # Perform hierarchical clustering
    linkage_matrix = linkage(data, method=linkage_method)

    # Plot the dendrogram
    dendrogram(linkage_matrix)
    plt.title('Agglomerative Clustering Dendrogram')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.show()
    plt.savefig("output_data/plot_image.png")
    st.image('output_data/plot_image.png')



def plot_kmeans_clusters(data, num_clusters = 2, max_iterations=100):
    # Use PCA for 2D visualization
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    # Plot the original data
    plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.5)
    plt.title('Original Data')
    plt.show()
    plt.savefig("output_data/plot_image.png")
    st.image('output_data/plot_image.png')

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters)
    for iteration in range(max_iterations):
        labels = kmeans.fit_predict(data)
        centroids_2d = pca.transform(kmeans.cluster_centers_)
        
        if iteration > 0 and np.array_equal(labels, prev_labels):
            print(f'Converged after {iteration + 1} iterations.')
            break

        # Save current labels for the next iteration
        prev_labels = labels.copy()
    
    # Plot the KMeans clusters
    clear_output(wait=True)
    plt.scatter(x=data_2d[:, 0], y=data_2d[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter(x=centroids_2d[:, 0], y=centroids_2d[:, 1], marker='X', s=200, c='red')
    plt.title(f'KMeans Clustering with {num_clusters} Clusters')
    plt.show()
    plt.savefig("output_data/plot_image.png")
    st.image('output_data/plot_image.png')

    return labels


# def cluster(clustering_model, clusternumber = 2, cluster_linkage="complete"):
#     model = SentenceTransformer('all-mpnet-base-v2')
#     df = pd.read_csv('output_data/stemmed_strings.csv')
#     corpus_embeddings = model.encode(df.stemmed_string)

#     if clustering_model == "KMeans":
#         # kmeans = KMeans(n_clusters=clusternumber, max_iter = 300, tol = 1e-4, algorithm = 'elkan')
#         # cluster_labels = kmeans.fit_predict(corpus_embeddings)
#         plot_kmeans_clusters(corpus_embeddings)
#     elif clustering_model == "Agglomerative":
#         # agglomerative = AgglomerativeClustering(linkage=cluster_linkage)
#         # cluster_labels = agglomerative.fit_predict(corpus_embeddings)
#         plot_agglomerative_dendrogram(corpus_embeddings)

#     else:
#         raise ValueError("Invalid clustering algorithm")

#     # Generate scatter plot
#     plt.title("Clustering Results")
#     plt.savefig("output_data/plot_image.png")

#     return cluster_labels


def search(input_keyword):
    model = SentenceTransformer('all-mpnet-base-v2')
    vector_of_input_keyword = model.encode(input_keyword)

    query = {
        "field": "ContentVector",
        "query_vector": vector_of_input_keyword,
        "k": 10,
        "num_candidates": 10000,                               
    }
    res = es.knn_search(index="all_content", knn=query, source=["title_article", "title", "content"])
    results = res["hits"]["hits"]

    return results

def main():
    model = SentenceTransformer('all-mpnet-base-v2')
    df = pd.read_csv('output_data/stemmed_strings.csv')
    corpus_embeddings = model.encode(df.stemmed_string)
    st.title("Choose your clustering algorithm")
    clustering_algorithm = st.selectbox("Clustering Algorithm", ["KMeans", "Agglomerative"])
    if clustering_algorithm == "Agglomerative":
        cluster_linkage = st.selectbox("Linkage", ["single", "complete", "average", "ward"])
    else :
        clusternumber = st.selectbox("Number of clusters", [2, 3, 4, 5, 6, 7, 8, 9, 10])
    if st.button("Cluster!"):
        if clustering_algorithm == "Agglomerative":
            plot_agglomerative_dendrogram(corpus_embeddings, cluster_linkage)
        else:
            plot_kmeans_clusters(corpus_embeddings, clusternumber)
    
    st.title("Search Articles")

    # Input: User enters search query
    search_query = st.text_input("Enter your search query")

    # Button: User triggers the search
    if st.button("Search"):
        if search_query:
            # Perform the search and get results
            # if clustering_algorithm == "Agglomerative":
            #     cluster_labels = cluster(clustering_algorithm, clusternumber, cluster_linkage)
            # else:
            #     cluster_labels = cluster(clustering_algorithm, clusternumber)

            results = search(search_query)

            # Display search results
            st.subheader("Search Results")
            for result in results:
                with st.container():
                    if '_source' in result:
                        try:
                            st.header(f"{result['_source']['title']}")
                        except Exception as e:
                            print(e)
                        
                        try:
                            st.caption(f"Similarity Score: {result['_score']}")
                        except Exception as e:
                            print(e)
                        
                        try:
                            st.write(f"Article: {result['_source']['title_article']}")
                        except Exception as e:
                            print(e)
                        
                        try:
                            st.write(f"Content: {result['_source']['content']}")
                        except Exception as e:
                            print(e)
                        st.divider()
                        


                    
if __name__ == "__main__":
    main()
