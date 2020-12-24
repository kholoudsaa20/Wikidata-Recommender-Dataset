
from collections import defaultdict
import csv
import numpy as np
import random
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer
import string
import pandas as pd

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize

from gensim.models import Word2Vec
from node2vec import Node2Vec

import tensorrec
from catboost.utils import get_roc_curve, eval_metric

from sys import getsizeof
import timeit

from statistics import mean
from sklearn.model_selection import KFold



#Step1: Loading and reading the data 
##Loading the interactions data 
print('Loading interaction data')
def csv_reader(file_name):
    with open(file_name, "r", encoding="utf8") as interactions_file:
        interactions_file_reader = csv.reader(interactions_file)
        keys = next(interactions_file_reader)
        for line in interactions_file_reader:
            yield list(line)

raw_interactions=[]
for i in csv_reader('Wikidata-Active-User-Subset.csv'):
    raw_interactions.append(i)


# Iterate through the input to map Wikidata IDs to new internal IDs
# The new internal IDs will be created by the defaultdict on insertion
wikidata_to_internal_user_ids = defaultdict(lambda: len(wikidata_to_internal_user_ids))
wikidata_to_internal_item_ids = defaultdict(lambda: len(wikidata_to_internal_item_ids))

def generator_fn(alist):
    for j in alist:
        yield j

for row in generator_fn(raw_interactions):
    row[0] = wikidata_to_internal_user_ids[row[0]]
    row[1] = wikidata_to_internal_item_ids[row[1]]
    row[2] = float(row[2])

n_users = len(wikidata_to_internal_user_ids)
n_items = len(wikidata_to_internal_item_ids)


print('n_users')
print(n_users)

print('n_items')
print(n_items)


##--------------------------------------------------------------------------------------------------------------------------    
##Loading the items' relationship features - graph features
    
###Load the trained node2vec model 
node_model = Word2Vec.load('D:\\Full_Dump\\word2vec_models\\s8_node2vec_0.25_0.25_10_100_100.model')

items_node_embeddings = []
for k, v in wikidata_to_internal_item_ids.items():
    if k in node_model.wv.vocab: 
        items_node_embeddings.append(node_model.wv.get_vector(k))
    else: 
        items_node_embeddings.append(np.zeros((300,), dtype=float))
        
item_relations_features_arr = np.stack(items_node_embeddings, axis=0)
#print(item_relations_features_arr.shape)

n_relations_features = item_relations_features_arr.shape[1]

# Coerce the item relations features to a sparse matrix, which TensorRec expects
item_relations_features_arr = sparse.coo_matrix(item_relations_features_arr)



##--------------------------------------------------------------------------------------------------------------------------
##Loading the items' contents 

###Load the trained word2vec model 
content_model = Word2Vec.load("D:\\Full_Dump\\Word2Vec_models\\all_wikidata_items_word2vec.model")


## 1-Removing stop words, converting lower case and punctuation from item's contents
def make_lower_case(text):
    return text.lower()

def remove_single_letter(text):
    return [w for w in text if len(w)>1]

def remove_spaces(text):
    return [word.strip() for word in text]

def clean_preprocess_topics(listText):
    cleaned = make_lower_case(listText)
    tokens = [word_tokenize(word) for word in cleaned.split()] #splits tokens into words based on white space
    tokens_flatten =[y for x in tokens for y in x]
    stops = set(stopwords.words("english"))
    no_stop = []
    for i in tokens_flatten:
        if i not in stops:
            no_stop.append(i)
    return no_stop

def remove_punctuation(text):
    return ''.join(c for c in text if c not in string.punctuation)

def remove_all_but_keep_noun_adj(text):
    #tokens = [word_tokenize(word) for word in text] #splits tokens into words based on white space and punctuation
    #stops = set(stopwords.words("english"))
    tagged = nltk.pos_tag(text)
    #tagged = [nltk.pos_tag(word) for word in text]
    #print(tagged)
    edited_set = [word for word, tag in tagged if tag != 'VB' and tag != 'VBD' and tag != 'VBG' and tag != 'VBN' 
                  and tag != 'VBP' and tag != 'VBZ' and tag != 'CC' and tag != 'CD' and tag != 'DT' and tag != 'EX' 
                  and tag != 'IN' and tag != 'LS' and tag != 'MD' and tag != 'PDT' and tag != 'POS' and tag != 'PRP'
                  and tag != 'PRP$' and tag != 'RB' and tag != 'RBR' and tag != 'RBS' and tag != 'RP' and tag != 'SYM'
                  and tag != 'TO' and tag != 'UH' and tag != 'WDT' and tag != 'WP' and tag != 'WP$' and tag != 'WRB']
    #print(edited_set)
    return edited_set

def clean_preprocess_contents(listText):
    cleaned = make_lower_case(listText)
    cleaned = remove_punctuation(cleaned)
    tokens = [word_tokenize(word) for word in cleaned.split()] #splits tokens into words based on white space
    tokens_flatten =[y for x in tokens for y in x]
    stops = set(stopwords.words("english"))
    no_stop = []
    for i in tokens_flatten:
        if i not in stops:
            no_stop.append(i)
    cleaned = remove_single_letter(no_stop)
    final_cleaned = remove_all_but_keep_noun_adj(cleaned)
    return final_cleaned

print('Loading item content')
def csv_content_reader(file_name):
    with open(file_name, "r", encoding="utf8") as items_file:
        item_file_reader = csv.reader(items_file)
        keys = next(item_file_reader)
        for line in item_file_reader:
            yield list(line)

raw_item_metadata=[]
for i in csv_content_reader('Item-Contents-of-Wikidata-19M.csv'):
    raw_item_metadata.append(i)

item_contents_by_internal_id = {}
for row in raw_item_metadata:
    row[0] = wikidata_to_internal_item_ids[row[0]]  # Map to IDs
    combined = row[1] + row[2]
    cleaned_combined = clean_preprocess_contents(combined)
    item_contents_by_internal_id[row[0]] = cleaned_combined


##Build a list of contents where the index is the internal item ID and
##the value is a list of item's contents (corpus)
item_contents = [item_contents_by_internal_id[internal_id] for internal_id in range(n_items)]


       
##get the items' content embeddings using the trained word2vec model - using max-pool - Option-1
count = 0
item_embeddings_1 = []
for i in item_contents:
    embeddings_single_item = []
    max_word2vec = None
    for word in i:
        if word in content_model.wv.vocab:
            word2vec = content_model[word]
            embeddings_single_item.append(word2vec)
            
    if len(embeddings_single_item)!=0:
            single_item_emb_arr = np.array(embeddings_single_item)
            max_word2vec = np.max(single_item_emb_arr, axis=0)
    else:
        #print('There is no embeddings appended in the list!!')
        count+=1
         
    if max_word2vec is not None:
        item_embeddings_1.append(max_word2vec)
    else:
        item_embeddings_1.append(np.zeros((300,), dtype=float))
           

##get the items' content embeddings using the trained word2vec model - using mean-pool - Option-2
# =============================================================================
# count = 0
# item_embeddings_2 = []
# for i in item_contents:
#     embeddings_single_item = []
#     max_word2vec = None
#     for word in i:
#         if word in content_model.wv.vocab:
#             word2vec = content_model[word]
#             embeddings_single_item.append(word2vec)
#             
#     if len(embeddings_single_item)!=0:
#             single_item_emb_arr = np.array(embeddings_single_item)
#             max_word2vec = np.mean(single_item_emb_arr, axis=0)
#     else:
#         print('There is no embeddings appended in the list!!')
#         count+=1
#          
#     if max_word2vec is not None:
#         item_embeddings_2.append(max_word2vec)
#     else:
#         item_embeddings_2.append(np.zeros((300,), dtype=float))
# 
# 
# =============================================================================


item_contents_features_arr = np.stack(item_embeddings_1, axis=0)

n_content_features = item_contents_features_arr.shape[1]
'''print("vectorized content example for item with content {}:\n{}".format(item_contents_by_internal_id[0], item_embeddings_array[0]))'''

# Coerce the item contents features to a sparse matrix, which TensorRec expects
item_contents_features_arr = sparse.coo_matrix(item_contents_features_arr)
#print(item_contents_features_arr.shape)



#Step2: Train-test sets preparation 
## Shuffle the interactions_data and split them into train/test sets 80%/20% in random way
random.shuffle(raw_interactions)  # Shuffles the list in-place
cutoff = int(.8 * len(raw_interactions))
train_set = raw_interactions[:cutoff]
test_set = raw_interactions[cutoff:]
validation_set = train_set[:0.2]


print('train_set')
print(len(train_set))
print('test_set')
print(len(test_set))



#Step3: Data conversion to sparse matrix
## This method converts a list of (user, item, frequence) to a sparse matrix
def interactions_list_to_sparse_matrix(interactions):
    users_column, items_column, ratings_column= zip(*interactions)
    return sparse.coo_matrix((ratings_column, (users_column, items_column)), shape=(n_users, n_items))

# Create sparse matrices of interaction data
sparse_train_set = interactions_list_to_sparse_matrix(train_set)
sparse_test_set = interactions_list_to_sparse_matrix(test_set)


def sort_coo(m):
    tuples = zip(m.row, m.col, m.data)
    return sorted(tuples, key=lambda x: (x[0], x[2]), reverse=True)


sparse_full_interactions = interactions_list_to_sparse_matrix(raw_interactions)



#Step4: Model fitting
## Construct indicator features for users and items
user_indicator_features = sparse.identity(n_users)
item_indicator_features = sparse.identity(n_items)

## Before calculating the metrics, we’ll want to decide which interactions should count as a “preferred” 
##In this case, I’ve chosen to use all interactions of at least 2.0 as “likes” and ignore the rest.
'''sparse_train_set_4plus = sparse_train_set.multiply(sparse_train_set >= 2.0)
sparse_test_set_4plus = sparse_test_set.multiply(sparse_test_set >= 2.0)'''

## This method consumes item ranks for each user and prints out recall@K train/test metrics
def evaluate_results(ranks, scores):
    train_precision_at_5 = tensorrec.eval.precision_at_k(test_interactions=sparse_train_set, predicted_ranks=ranks, k=5).mean()
    test_precision_at_5 = tensorrec.eval.precision_at_k(test_interactions=sparse_test_set, predicted_ranks=ranks, k=5).mean()
    print("Precision at {}: Train: {:.4f} Test: {:.4f}".format(5, train_precision_at_5, test_precision_at_5))
    
    train_precision_at_10 = tensorrec.eval.precision_at_k(test_interactions=sparse_train_set, predicted_ranks=ranks, k=10).mean()
    test_precision_at_10 = tensorrec.eval.precision_at_k(test_interactions=sparse_test_set, predicted_ranks=ranks, k=10).mean()
    print("Precision at {}: Train: {:.4f} Test: {:.4f}".format(10, train_precision_at_10, test_precision_at_10))
    
    train_recall_at_50 = tensorrec.eval.recall_at_k(test_interactions=sparse_train_set, predicted_ranks=ranks, k=50).mean()
    test_recall_at_50 = tensorrec.eval.recall_at_k(test_interactions=sparse_test_set, predicted_ranks=ranks, k=50).mean()
    print("Recall at {}: Train: {:.4f} Test: {:.4f}".format(50, train_recall_at_50, test_recall_at_50))
    
    train_recall_at_100 = tensorrec.eval.recall_at_k(test_interactions=sparse_train_set, predicted_ranks=ranks, k=100).mean()
    test_recall_at_100 = tensorrec.eval.recall_at_k(test_interactions=sparse_test_set, predicted_ranks=ranks, k=100).mean()
    print("Recall at {}: Train: {:.4f} Test: {:.4f}".format(100, train_recall_at_100, test_recall_at_100))
    
    train_recall_at_200 = tensorrec.eval.recall_at_k(test_interactions=sparse_train_set, predicted_ranks=ranks, k=200).mean()
    test_recall_at_200 = tensorrec.eval.recall_at_k(test_interactions=sparse_test_set, predicted_ranks=ranks, k=200).mean()
    print("Recall at {}: Train: {:.4f} Test: {:.4f}".format(200, train_recall_at_200, test_recall_at_200))
    
    train_recall_at_1000 = tensorrec.eval.recall_at_k(test_interactions=sparse_train_set, predicted_ranks=ranks, k=1000).mean()
    test_recall_at_1000 = tensorrec.eval.recall_at_k(test_interactions=sparse_test_set, predicted_ranks=ranks, k=1000).mean()
    print("Recall at {}: Train: {:.4f} Test: {:.4f}".format(1000, train_recall_at_1000, test_recall_at_1000))
    
    '''train_ndcg_at_K = tensorrec.eval.ndcg_at_k(test_interactions=sparse_train_set, predicted_ranks=ranks, k=rk).mean()
    test_ndcg_at_K = tensorrec.eval.ndcg_at_k(test_interactions=sparse_test_set, predicted_ranks=ranks, k=rk).mean()
    print("NDCG at {}: Train: {:.4f} Test: {:.4f}".format(rk, train_ndcg_at_K, test_ndcg_at_K))'''
    
    auc = eval_metric(sparse_full_interactions.data, scores, 'AUC:type=Ranking', group_id=sparse_full_interactions.row)
    print('Ranking ROC AUC = {0:.2f}'.format(auc))





## Fit a collaborative filetring model 
# =============================================================================
# print("Training collaborative filter")
# ranking_cf_model = tensorrec.TensorRec(n_components=300, loss_graph=tensorrec.loss_graphs.WMRBLossGraph())
# ranking_cf_model.fit(interactions=sparse_train_set,
#                      user_features=user_indicator_features,
#                      item_features=item_indicator_features,
#                      n_sampled_items=int(n_items * .01))
# 
# 
# # Check the results of the WMRB MF CF model
# print("\n WMRB matrix factorization collaborative filter:")
# predicted_ranks = ranking_cf_model.predict_rank(user_features=user_indicator_features, item_features=item_indicator_features)
# check_results(predicted_ranks)
# 
# =============================================================================


## Fit a content-based model using the relations/contents or all as item features
# =============================================================================
# '''combined_item_features = sparse.hstack([item_contents_features_arr, item_relations_features_arr])
# Num_components = n_content_features + n_relations_features'''
# 
# '''print("Training content-based recommender using items' contents")
# content_model = tensorrec.TensorRec(n_components=n_content_features,
#                                     item_repr_graph=tensorrec.representation_graphs.FeaturePassThroughRepresentationGraph(),
#                                     loss_graph=tensorrec.loss_graphs.WMRBLossGraph())'''
# 
# '''print("Training content-based recommender using items' relations")
# content_model = tensorrec.TensorRec(n_components=n_graph_features,
#                                     item_repr_graph=tensorrec.representation_graphs.FeaturePassThroughRepresentationGraph(),
#                                     loss_graph=tensorrec.loss_graphs.WMRBLossGraph())'''
# 
# '''print("Training content-based recommender using item's contents and relations")
# content_model = tensorrec.TensorRec(n_components=Num_components,
#                                     item_repr_graph=tensorrec.representation_graphs.FeaturePassThroughRepresentationGraph(),
#                                     loss_graph=tensorrec.loss_graphs.WMRBLossGraph())'''
# 
# content_model.fit(interactions=sparse_train_set,
#                   user_features=user_indicator_features,
#                   item_features=item_contents_features_arr,
#                   n_sampled_items=int(n_items * .01))
# 
# ##Check the results of the content-based model
# print("\n Content-based recommender:")
# predicted_ranks = content_model.predict_rank(user_features=user_indicator_features, item_features=item_contents_features_arr)
# check_results(predicted_ranks)
# 
# =============================================================================

## Concatenating the item-related features to the indicator features for a hybrid recommender system   
'''full_item_features = sparse.hstack([item_indicator_features, item_graph_features_arr])'''
'''full_item_features = sparse.hstack([item_indicator_features, item_embeddings_array])'''

full_item_features = sparse.hstack([item_indicator_features, item_contents_features_arr, item_relations_features_arr])
'''full_item_features = sparse.hstack([item_indicator_features, item_topic_features, item_embeddings_array])'''
'''full_item_features = sparse.hstack([item_indicator_features, item_topic_features, item_graph_features_arr])'''
'''full_item_features = sparse.hstack([item_indicator_features, item_topic_features, item_embeddings_array, item_graph_features_arr])'''

print("Training hybrid recommender using Hybrid of CF data and item-based features")
hybrid_model = tensorrec.TensorRec(n_components=300, loss_graph=tensorrec.loss_graphs.WMRBLossGraph())
hybrid_model.fit(interactions=sparse_train_set,
                 user_features=user_indicator_features,
                 item_features=full_item_features,
                 n_sampled_items=int(n_items *.01))

print("Hybrid recommender:")
predicted_ranks = hybrid_model.predict_rank(user_features=user_indicator_features, item_features=full_item_features)
predicted_scores = hybrid_model.predict(user_features=user_indicator_features, item_features=full_item_features)

evaluate_results(predicted_ranks, predicted_scores)


'''print(type(predicted_ranks))
print(predicted_ranks[0])
print(len(predicted_ranks[0])) #Number of items 
print(len(predicted_ranks)) #Number of users'''



