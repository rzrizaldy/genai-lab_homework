import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import llm_functions
from llm_functions import LLM_requesters

llm = LLM_requesters()


class ragLLM:

    def __init__(self):
        pass

    def get_rag_index(self, rag_sentences, sentence_embeds):
        d = sentence_embeds.shape[1]
        nb = len(set(rag_sentences))
        nq = 10000 
        np.random.seed(1234)             # make reproducible
        xb = np.random.random((nb, d)).astype('float32')
        nlist = 100
        index = faiss.IndexFlatL2(d)
        index
        index.add(sentence_embeds)
        index.train(sentence_embeds)
        print(index.is_trained)

    #def get_sys_message(self, user_prompt, k, rag_sentences, sentence_embeds):
       # index = self.get_rag_index(rag_sentences, sentence_embeds)
       # model = SentenceTransformer('bert-base-nli-mean-tokens')
       # k=k
       # xq = model.encode([user_prompt])
       # D, I = index.search(xq, k)  # search
       # first_index = I[0]  # Get the first index from I
        #rag_search_results = sentences_df['sentence'].iloc[first_index].sum()
       # return rag_search_results
        
    def rag_llm(self, model, k, user_prompt, task, system_message):
        #f=self.get_sys_message(user_prompt, k, rag_sentences, sentence_embeds)
        response=llm.get_llm_response(model, user_prompt, system_message, task)
        return response

    def rag_llm_openai(self, model, k, user_prompt, task, system_message ):
        import openai
        from openai import OpenAI
        client = OpenAI()
        
        #f=self.get_sys_message(user_prompt, k, rag_sentences, sentence_embeds)
        
        response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"Instruction: use the information in document {system_message} to perform the task in {task}."},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": system_message},
            {"role": "user", "content": task}
        ]
        )
        return response.choices[0].message.content

    
    
        