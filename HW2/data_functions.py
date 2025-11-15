import huggingface_hub, datasets, transformers, os, sys, json, random, re, nltk, time, requests, faiss
import pandas as pd
import numpy as np

from transformers import pipeline
from sentence_transformers import SentenceTransformer

from nltk.tokenize import sent_tokenize

from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt_tab')

class DataFunctions:

    def __init__(self):
        pass

    def split_resumes_to_sentences(self, df, text_column):
        """
        Split the resumes into individual sentences and assign unique identifiers.
        
        Parameters:
            df (pd.DataFrame): The DataFrame containing the resumes.
            text_column (str): The name of the column containing the resume texts.
            
        Returns:
            pd.DataFrame: A DataFrame with each sentence and its corresponding unique identifier.
        """
        # Initialize an empty list to hold the resulting data
        sentences_list = []
        
        # Iterate through the DataFrame rows
        for idx, row in df.iterrows():
            # Tokenize the resume text into sentences
            sentences = sent_tokenize(row[text_column])
            
            # Append each sentence along with the original index to the list
            for sentence in sentences:
                sentences_list.append((idx, sentence))
        
        # Convert the list to a DataFrame
        sentences_df = pd.DataFrame(sentences_list, columns=['unique_identifier', 'sentence'])
        
        return sentences_df
    
    
    # Splitting Text into Sentences
    def split_text_into_sentences(self, text):
        sentences = nltk.sent_tokenize(text)
        return sentences
    
    def split_text_into_sentences(self, text):
        sentences = sent_tokenize(text, language='english')  # Default is usually 'english'
        return sentences
    
    def get_sys_message(self, q, k):
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        k=k
        xq = model.encode([q])
        D, I = index.search(xq, k)  # search
        first_index = I[0]  # Get the first index from I
        first_row_string = sentences_df['sentence'].iloc[first_index].sum()
        return first_row_string 