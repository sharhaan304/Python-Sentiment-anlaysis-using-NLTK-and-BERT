#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().system('pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121')


# In[6]:


get_ipython().system('pip install transformers requests beautifulsoup4 pandas numpy')


# In[9]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re


# In[11]:


tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


# In[24]:


tokens = tokenizer.encode('oh my god, Amazing', return_tensors='pt')


# In[25]:


tokens


# In[26]:


result = model(tokens)


# In[27]:


result.logits


# In[28]:


int(torch.argmax(result.logits))+1


# In[29]:


r = requests.get('https://www.yelp.com/biz/social-brew-cafe-pyrmont')
soup = BeautifulSoup(r.text, 'html.parser')
regex = re.compile('.*comment.*')
results = soup.find_all('p', {'class':regex})
reviews = [result.text for result in results]


# In[33]:


reviews[0]


# In[35]:


import pandas as pd
import numpy as np


# In[36]:


df = pd.DataFrame(np.array(reviews), columns=['review'])


# In[44]:


df['review'].iloc[1]


# In[41]:


def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    results = model(tokens)
    return int(torch.argmax(results.logits))+1


# In[43]:


sentiment_score(df['review'].iloc[1])


# In[45]:


df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512]))


# In[46]:


df


# In[47]:


df['review'].iloc[6]


# In[ ]:




