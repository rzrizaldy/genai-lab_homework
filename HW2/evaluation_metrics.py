from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import llm_functions

class RagMetrics:
    def __init__(self):
        pass
        
    def compare_text_similarity(text_a, text_b, text_c):
        """
        Compares the similarity between three texts using TF-IDF vectors and cosine similarity.
        
        Parameters:
        - text_a (str): Text A
        - text_b (str): Text B
        - text_c (str): Text C
        
        Returns:
        - A dictionary with similarity scores between Text A & Text B, A & C, and B & C.
        """
        # Initialize the vectorizer and transform texts into TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b, text_c])
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Similarity between Text A and B, A and C, and then B and C
        similarity_scores = {
            "A_B": similarity_matrix[0, 1],
            "A_C": similarity_matrix[0, 2],
            "B_C": similarity_matrix[1, 2]
        }
    
        return similarity_scores
    
    def compare_text_similarity_response2context(om, model_response):
        """
        Compares the similarity between three texts using TF-IDF vectors and cosine similarity.
        
        Parameters:
        - text_a (str): Text A
        - text_b (str): Text B
        - text_c (str): Text C
        
        Returns:
        - A dictionary with similarity scores between Text A & Text B, A & C, and B & C.
        """
        # Initialize the vectorizer and transform texts into TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([om, model_response])
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Similarity between Text A and B, A and C, and then B and C
        similarity_scores = {
            "A_B": similarity_matrix[0, 1],
            #"A_C": similarity_matrix[0, 2],
            #"B_C": similarity_matrix[1, 2]
        }
    
        return similarity_scores