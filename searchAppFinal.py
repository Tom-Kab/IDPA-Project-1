import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

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
