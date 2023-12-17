import pandas as pd
import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt


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


def cluster(clustering_model):
    model = SentenceTransformer('all-mpnet-base-v2')
    df = pd.read_csv('output_data/stemmed_strings.csv')
    corpus_embeddings = model.encode(df.stemmed_string)

    if clustering_model == "KMeans":
        kmeans = KMeans(n_clusters=5)
        cluster_labels = kmeans.fit_predict(corpus_embeddings)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x')
    elif clustering_model == "Agglomerative":
        agglomerative = AgglomerativeClustering(n_clusters=5)
        cluster_labels = agglomerative.fit_predict(corpus_embeddings)
    else:
        raise ValueError("Invalid clustering algorithm")

    # Generate scatter plot
    plt.scatter(corpus_embeddings[:, 0], corpus_embeddings[:, 1], c=cluster_labels)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Clustering Results")
    plt.savefig("output_data/plot_image.png")

    return cluster_labels


def search(input_keyword):
    model = SentenceTransformer('all-mpnet-base-v2')
    vector_of_input_keyword = model.encode(input_keyword)

    query = {
        "field": "ContentVector",
        "query_vector": vector_of_input_keyword,
        "k": 10,
        "num_candidates": 10000
    }
    res = es.knn_search(index="all_content", knn=query, source=["title_article", "title", "content"])
    results = res["hits"]["hits"]

    return results

def main():
    st.title("Choose your clustering algorithm")
    clustering_algorithm = st.selectbox("Clustering Algorithm", ["KMeans", "Agglomerative"])
    if st.button("Cluster!"):
        cluster_labels = cluster(clustering_algorithm)
        st.image('output_data/plot_image.png')
    
    
    
    
    st.title("Search Articles")

    # Input: User enters search query
    search_query = st.text_input("Enter your search query")

    # Button: User triggers the search
    if st.button("Search"):
        if search_query:
            # Perform the search and get results
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
