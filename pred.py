import pandas as pd
import numpy as np
import re
from collections import Counter

STOP_WORDS = {'call', 'upon', 'still', 'nevertheless', 'down', 'every', 'forty', 're', 'always', 'whole', 'side', "n't", 
              'now', 'however', 'an', 'show', 'least', 'give', 'below', 'did', 'sometimes', 'which', "'s", 'nowhere', 'per', 
              'hereupon', 'yours', 'she', 'moreover', 'eight', 'somewhere', 'within', 'whereby', 'few', 'has', 'so', 'have', 
              'for', 'noone', 'top', 'were', 'those', 'thence', 'eleven', 'after', 'no', 'll', 'others', 'ourselves', 'themselves', 
              'though', 'that', 'nor', 'just', 's', 'before', 'had', 'toward', 'another', 'should', 'herself', 'and', 'these', 
              'such', 'elsewhere', 'further', 'next', 'indeed', 'bottom', 'anyone', 'his', 'each', 'then', 'both', 'became', 
              'third', 'whom', 've', 'mine', 'take', 'many', 'anywhere', 'to', 'well', 'thereafter', 'besides', 'almost', 
              'front', 'fifteen', 'towards', 'none', 'be', 'herein', 'two', 'using', 'whatever', 'please', 'perhaps', 'full', 
              'ca', 'we', 'latterly', 'here', 'therefore', 'how', 'was', 'made', 'the', 'or', 'may', 're', 'namely', 
              "'ve", 'anyway', 'amongst', 'used', 'ever', 'of', 'there', 'than', 'why', 'really', 'whither', 'in', 'only', 
              'wherein', 'last', 'under', 'own', 'therein', 'go', 'seems', 'm', 'wherever', 'either', 'someone', 'up', 
              'doing', 'on', 'rather', 'ours', 'again', 'same', 'over', 's', 'latter', 'during', 'done', "'re", 'put', 
              "'m", 'much', 'neither', 'among', 'seemed', 'into', 'once', 'my', 'otherwise', 'part', 'everywhere', 'never', 
              'myself', 'must', 'will', 'am', 'can', 'else', 'although', 'as', 'beyond', 'are', 'too', 'becomes', 'does', 'a', 
              'everyone', 'but', 'some', 'regarding', 'll', 'against', 'throughout', 'yourselves', 'him', "'d", 'it', 'himself', 
              'whether', 'move', 'm', 'hereafter', 're', 'while', 'whoever', 'your', 'first', 'amount', 'twelve', 'serious', 
              'other', 'any', 'off', 'seeming', 'four', 'itself', 'nothing', 'beforehand', 'make', 'out', 'very', 'already', 
              'various', 'until', 'hers', 'they', 'not', 'them', 'where', 'would', 'since', 'everything', 'at', 'together', 
              'yet', 'more', 'six', 'back', 'with', 'thereupon', 'becoming', 'around', 'due', 'keep', 'somehow', 'nt', 'across', 
              'all', 'when', 'i', 'empty', 'nine', 'five', 'get', 'see', 'been', 'name', 'between', 'hence', 'ten', 'several', 'from', 
              'whereupon', 'through', 'hereby', "'ll", 'alone', 'something', 'formerly', 'without', 'above', 'onto', 'except', 'enough', 
              'become', 'behind', 'd', 'its', 'most', 'nt', 'might', 'whereas', 'anything', 'if', 'her', 'via', 'fifty', 'is', 
              'thereby', 'twenty', 'often', 'whereafter', 'their', 'also', 'anyhow', 'cannot', 'our', 'could', 'because', 'who', 
              'beside', 'by', 'whence', 'being', 'meanwhile', 'this', 'afterwards', 'whenever', 'mostly', 'what', 'one', 'nobody', 
              'seem', 'less', 'do', 'd', 'say', 'thus', 'unless', 'along', 'yourself', 'former', 'thru', 'he', 'hundred', 'three', 
              'sixty', 'me', 'sometime', 'whose', 'you', 'quite', 've', 'about', 'even'}


"""
Helper function for organize_data() to replace NaN values with column means.
"""
def replace_nan(df, column_means):
  columns_to_replace = ['Q1', 'Q2', 'Q3', 'Q4', 'Q7', 'Q8', 'Q9']
  
  for index, row in df.iterrows():
    for col in columns_to_replace:
      if pd.isnull(row[col]):
        df.at[index, col] = int(column_means[col])

  return df


"""
Organize the data into a pandas matrix, for easy calculations.
"""
def organize_data(df):
  df["Q5"] = df["Q5"].fillna('')
  df["Q6"] = df["Q6"].fillna('')
  df["Q10"] = df["Q10"].fillna('')

  # Convert Q5 categories into binary features
  q5_categories = ["Partner", "Friends", "Siblings", "Co-worker"]
  for category in q5_categories:
    df[f"Q5_{category}"] = df["Q5"].apply(lambda s: 1 if category in s else 0)

  # Convert Q6 categories into binary features
  q6_categories = ["Skyscrapers", "Sport", "Art and Music", "Carnival", "Cuisine", "Economic"]
  for category in q6_categories:
    numbers = [int(re.findall(f"{category}=>(\d+)", s)[0]) if re.findall(f"{category}=>(\d+)", s) else 0 for s in
                df["Q6"]]
    df[f"Q6_{category}"] = numbers

  # Drop original columns
  df.drop(["Q5", "Q6"], axis=1, inplace=True)

  # fix string values input to Q7 and Q9
  # df['Q7'] = pd.to_numeric(df['Q7'].str.replace(',', ''), errors='coerce')
  # df['Q9'] = pd.to_numeric(df['Q9'].str.replace(',', ''), errors='coerce')
  df['Q7'] = df['Q7'].apply(convert_to_numeric)
  df['Q9'] = df['Q9'].apply(convert_to_numeric)


  df["Q10"] = df["Q10"].astype(str)
  df["Q10"] = df["Q10"].apply(lambda s: re.sub(r'[^a-zA-Z\s]', ' ', s))
  df["Q10"] = df["Q10"].apply(lambda s: ' '.join(s.split()))

  vocab_list = list()
  for text in df["Q10"]:
    words = text.split()
    vocab_list.extend(words)
  vocab_list = list(set(word.lower() for word in vocab_list))

  filtered_vocab_list = [word for word in vocab_list if word.lower() not in STOP_WORDS]

  for word in filtered_vocab_list:
    df[word] = df["Q10"].apply(lambda x: Counter(x.split())[word])

  column_means = df[["Q1", "Q2", "Q3", "Q7", "Q8", "Q9", "Q4"]].mean().astype(int)
  df = replace_nan(df, column_means)
  df.drop(["Q10"], axis=1, inplace=True)

  return df

def convert_to_numeric(x):
  try:
    if isinstance(x, float):
      return (x)
    if isinstance(x, str):
      return int(x.replace(',', '')) 
    else:
      return int(x)

  except ValueError:
    return int(float(x))


"""
Helper function to make a prediction for vector x.
Uses a decision tree based model for this prediction.
"""
def predict(x):
  if x['Q7'] <= 15.5:
    if x['Q6_Skyscrapers'] <= 4.5:
      if x['Q6_Economic'] <= 4.5:
        if x['Q6_Cuisine'] <= 3.5:
          if x['Q5_Partner'] <= 0.5:
            return 'Rio de Janeiro'
          else:
            return 'Paris'
        else:
          return 'Paris'
      else:
        if x['Q6_Cuisine'] <= 5.5:
          return 'New York City'
        else:
          if x['Q6_Art and Music'] <= 3.5:
            return 'New York City'
          else:
            return 'Paris'
    else:
      if x['Q7'] <= 8.5:
        if x['Q2'] <= 3.5:
          if x['Q6_Economic'] <= 4.5:
            return 'Paris'
          else:
            return 'New York City'
        else: 
          return 'New York City'
      else:
        if x['Q2'] <= 4.5:
          if x['Q6_Economic'] <= 4.5:
            return 'Paris'
          else:
            return 'Dubai'
        else:
          return 'New York City'
  else:
    if x['Q6_Skyscrapers'] <= 4.5:
      if x['Q6_Carnival'] <= 4.5:
        if x['Q1'] <= 4.5:
          if x['Q6_Sport'] <= 3.5:
            return 'Dubai'
          else:
            return 'Rio de Janeiro'
        else:
          if x['Q7'] <= 23.5:
            return 'Paris'
          else:
            return 'Rio de Janeiro'
      else:
        if x['Q4'] <= 2.5:
          if x['Q6_Cuisine'] <= 2.5:
            return 'Rio de Janeiro'
          else:
            return 'Dubai'
        else:
          return 'Rio de Janeiro'
    else:
      if x['Q2'] <= 3.5:
        if x['Q4'] <= 4.5:
          return 'Dubai'
        else: 
          if x['Q6_Economic'] <= 5.5:
            return 'Rio de Janeiro'
          else:
            return 'Dubai'
      else:
        if x['Q6_Art and Music'] <= 4.5:
          if x['Q6_Sport'] <= 5.5:
            return 'Dubai'
          else:
            return 'Paris'
        else:
          if x['Q3'] <= 4.5:
            return 'Paris'
          else:
            return 'Dubai'


"""
Make predictions for the data in filename.
"""

def predict_all(filename):
  df = pd.read_csv(filename)
  X = organize_data(df)

  predictions = list(X.apply(predict, axis=1))

  return predictions

  