import nltk
nltk.download('brown')
nltk.download('universal_tagset')
import numpy as np
from nltk.corpus import brown
import collections
import re
import sys
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
import csv
import itertools
from itertools import groupby


############ creating the set rare words that occur less than 5 times ############################
rare_words = set()
### tag dictionary global variable ###
tag_dict = {}
## feature dictionary global variable ###
feature_dict = {}

def word_ngram_features(i, words):
  ''' The argument i (int) is the index of the current word to generate features for, and words is a list containing all the words 
      in the sentence (ie. words[i] is the current word wi). The function should return a list containing the following features:
      • ‘prevbigram-[x]’, where [x] is the previous word wi−1
      • ‘nextbigram-[x]’, where [x] is the next word wi+1
      • ‘prevskip-[x]’, where [x] is the next previous word wi−2
      • ‘nextskip-[x]’, where [x] is the next next word wi+2
      • ‘prevtrigram-[x]-[y]’, where [x] is wi−1 and [y] is wi−2
      • ‘nexttrigram-[x]-[y]’, where [x] is wi+1 and [y] is wi+2
      • ‘centertrigram-[x]-[y]’, where [x] is wi−1 and [y] is wi+1
      You will need to check for corner cases where i ± 1 and/or i ± 2 are n '''
  list_ngram_features = list()
  list_len = len(words)
  
  #### checking for prevbigram  #######
  prevbigram = ""
  if i == 0:
    prevbigram = "<s>"
  else :
    prevbigram = words[i-1]
  list_ngram_features.append("prevbigram" + "-" + prevbigram)
  
  ### checking the next bigram ########
  nextbigram = ""
  if i+1 == list_len:
    nextbigram = "</s>"
  else:
    nextbigram = words[i+1]
    
  list_ngram_features.append("nextbigram" + "-" + nextbigram)
  
  ### checking for the prev skip ######
  prevskip = ""
  if i < 2:
    prevskip = "<s>"
  else:
    prevskip = words[i-2]
  
  list_ngram_features.append("prevskip" + "-" + prevskip)
  
  ### checking fro the next skip ######
  nextskip = ""
  if i+2 >= len(words):
    nextskip = "</s>"
  else:
    nextskip = words[i+2]
    
  list_ngram_features.append("nextskip" + "-" + nextskip)
  
  ### checking for prevtrigram #######
  prevtrigram1 = ""
  prevtrigram2 = ""
  if i == 0:
    prevtrigram1 = "<s>"
    prevtrigram2 = "<s>"
  elif i == 1:
    prevtrigram1 = words[i-1]
    prevtrigram2 = "<s>"
  else:
    prevtrigram1 = words[i-1]
    prevtrigram2 = words[i-2]
  
  list_ngram_features.append("prevtrigram" + "-" + prevtrigram1 + "-" + prevtrigram2)
  
  ### checking for nexttrigram ######
  nexttrigram1 = ""
  nexttrigram2 = ""
  if (i + 1) >= len(words):
    nexttrigram1 = "</s>"
    nexttrigram2 = "</s>"
  elif (i+1) >= len(words) - 1:
    nexttrigram1 = words[i+1]
    nexttrigram2 = "</s>"
  else:
    nexttrigram1 = words[i+1]
    nexttrigram2 = words[i+2]
    
  list_ngram_features.append("nexttrigram" + "-" + nexttrigram1 + "-" + nexttrigram2)
  
  ### checking for center trigram ###
  centertrigram1 = ""
  centertrigram2 = ""
  if len(words) == 1:
    centertrigram1 = "<s>"
    centertrigram2 = "</s>"
  elif i == 0:
    centertrigram1 = "<s>"
    centertrigram2 = words[i+1]
  elif (i+1) >= len(words):
    centertrigram1 = words[i-1]
    centertrigram2 = "</s>"
  else:
    centertrigram1 = words[i-1]
    centertrigram2 = words[i+1]
  
  list_ngram_features.append("centertrigram" + "-" + centertrigram1 + "-" + centertrigram2)
  
  return list_ngram_features
  

def word_features(word, rare_words):
  ''' The argument word is the current word wi , and rare words is the set of rare words we made in the previous section.
      The function should return a list containing the following features:
      • ‘word-[word]’, only if word is not in rare words
      • ‘capital’, only if word is capitalized
      • ‘number’, only if word contains a digit
      • ‘hyphen’, only if word contains a hyphen
      • ‘prefix[j]-[x]’, where [j] ranges from 1 to 4 and [x] is the substring containing
      the first [j] letters of word 
      ‘suffix[j]-[x]’, where [j] ranges from 1 to 4 and [x] is the substring containing
      the last [j] letters of word '''
  
  word_features_list = list()
 
  ### check if word not in rare word ###
  if word not in rare_words:
    word_features_list.append(word + "-" + word)
    
  ### check if word is capitalized ###
  if word[0].isupper():
    word_features_list.append("capital")
    
  ### check if word contains digit ####
  if re.search("\d", word):
    word_features_list.append("number")
    
  ### check if word contains hyphen ###
  if '-' in word:
    word_features_list.append("hyphen")
    
  ### prefix[j]-[x] where j ranges from 1 to 4 and [x] is the substring containing the first [j] letters of word ###
  
  j = 1
  
  while(j <= 4) and j <= len(word):
    word_features_list.append("prefix" + str(j) + "-" + word[0:j])
    j = j + 1
  
 
  ### suffix[j] - [x] where j ranges from 1 to 4 and [x] is the substring containing the last [j] letters of word ###
  j = 1
  while(j <= 4 and j <= len(word)):
    word_features_list.append("suffix" + str(j) + "-" + word[-1*j:])
    j = j + 1
  
  
  return word_features_list
    
  
def get_features(i, words, prevtag, rare_words):
  ''' training features[0][0] should be the list of features for the first word in the first
      sentence of the corpus. You will need to check for corner cases where i = 0 and there is
      no prevtag; in these cases, use the meta-tag ‘<S>’ .'''
  
  list_word_ngram_features = word_ngram_features(i,words)

  list_word_features = word_features(words[i],rare_words)
  
  features = list_word_ngram_features + list_word_features
  
  ### adding the tagbigram-[prevtag] to the final feature list ######
  features.append("tagbigram" + "-" + prevtag)

  ########## extra credit part for the HW3. Question 6 ##############
  features.append("word-" + words[i] + "-prevtag-" + prevtag)
  
  #### converting the features to lower case #########
  lower_features = list(map(lambda x:x.lower(),features))

  ### the below features do not need to be converted to lowercase ###########

  #### check all caps ######
  if words[i].isupper():
      lower_features.append("allcaps")
  
  ####### wordshape feature #########
  abstract_version = ""
  for character in words[i]:
      if character.islower():
          abstract_version = abstract_version + "x"
      elif character.isupper():
          abstract_version = abstract_version + "X"
      elif character.isdigit():
          abstract_version = abstract_version + "d"

  if len(abstract_version) > 0:
    lower_features.append("wordshape" + "-" + abstract_version)
    
  ### short wordshape feature #######
  short_word_shape = ''.join(ch for ch, _ in itertools.groupby(abstract_version))

  if len(short_word_shape) > 0:
    lower_features.append("short-wordshape-" + short_word_shape)

  #### all caps , atleast 1 hyphen, atleast 1 digit #####
  if words[i].isupper() and '-' in words[i] and  any(ch.isdigit() for ch in words[i]):
      lower_features.append("allcaps-digit-hyphen")
  
  ######## capital-followed by co feature #########

  if words[i][0].isupper():
      for indice in range(i+1,i+3):
          if indice < len(words):
            if words[indice] == "Co." or words[indice] == "Inc.":
                lower_features.append("capital-followedby-co")
                break
            else:
                break

  return lower_features

def remove_rare_features(features, n):
  ''' The argument features is a list of lists of feature lists (ie. training features), and n is the number of times a 
      feature must occur in order for us to keep it in the feature set '''
  ### creating a vocabulary that will store the feature and its count ####
  feature_vocabulary = {}
  
  for sentence_feature in features:
    for word_features in sentence_feature:
      for feature in word_features:
        feature_vocabulary[feature] = feature_vocabulary.get(feature,0) + 1
                           
  ###### creating two sets for rare and non rare feature vocab #############
  rare_features = set()
  non_rare_features = set()
  
  for feature,feature_count in feature_vocabulary.items():
    if feature_count < n:
      rare_features.add(feature)
    else:
      non_rare_features.add(feature)
      
  ########## removing the rare features from the training features set ############
  updated_training_features = list()

  for sentence_feature in features:
      word_features_list = list()
      for word_features in sentence_feature:
          word_list = list()
          for word_feature in word_features:
              if word_feature not in rare_features:
                  word_list.append(word_feature)
          word_features_list.append(word_list)
      updated_training_features.append(word_features_list)        

  return (updated_training_features,non_rare_features)


def build_Y(tags):
  ''' build train_Y for training data and return numpy array '''
  Y = list()

  for sentence_tag in tags:
    for word_tag in sentence_tag:
      Y.append(tag_dict[word_tag])

  return np.array(Y)

def build_X(features):
  ''' construct a sparse matrix using three lists, row, col, and value, to indicate which positions are not 0: 
        X[row[i][col[i]] = value[i], and all other positions in X are assumed to be 0. Of course, all non-0 positions 
        have value 1 for us, so value would just be a list of 1s '''
  examples = []
  feature_index = []
  i=0
  #### some theory here #####

  #### here , feature refers to the individual features for each word. So , for every word , we have a feature list 
  #### we append the number of times index i to the examples list for each word feature in word feature list. since we 
  #### want to keep track of count of features in each word. so if a word has features like [A,B,C] , we will append, index i 
  ### to the example list , 3 times.
  for sentence_features in features:
    for word_features in sentence_features:
      for feature in word_features:
        if feature in feature_dict:
          examples.append(i)
          feature_index.append(feature_dict[feature])
        
      i += 1

  values = [1] * len(examples)

  examples = np.asarray(examples)
  feature_index = np.asarray(feature_index)
  values = np.asarray(values)

  sparse_matrix = csr_matrix((values, (examples, feature_index)), shape = (i, len(feature_dict)), dtype = np.int8)
  return sparse_matrix

def viterbi(Y_start, Y_pred):
    
    N = np.shape(Y_pred)[0] + 1
    T = len(tag_dict)
    V = np.empty((N, T))
    BP = np.empty((N, T))

    for j in range(T):
        V[0][j] = Y_start[0][j]
        BP[0][j] = -1

    for i, row in enumerate(Y_pred):
            for k in range(T):
                sum = V[i, :] + Y_pred[i, :, k]
                V[i + 1, k] = max(sum)
                BP[i + 1, k] = int(np.argmax(sum))
    backward_indices = []
    index = np.argmax(V[N-1])
    backward_indices.append(index)
    for n in range(N - 1, 0, -1):
        index = BP[n, int(index)]
        backward_indices.append(index)
    
    for key,value in tag_dict.items():
        i = 0
        for bindex in backward_indices:
            if bindex == value:
                backward_indices[i] = key
            i = i + 1
    
    
    return list(reversed(backward_indices))

def load_test(filename):
    ''' The function should return a list of lists, where each sublist is a sentence (a line in the test file), and each item
        in the sentence is a word. '''
    final_list = list()

    with open(filename, encoding='utf-8', errors='ignore') as file:
        file_reader = csv.reader(file, delimiter='\t')
        for row in file_reader:
            updated_list = list()
            for word in row[0].split():
                updated_list.append(word.strip())            
            final_list.append(updated_list)
    
    return final_list 

def get_predictions(test_sentence, model):
    ''' The argument test sentence is a list containing a single list of words (we continue to use a nested list
        because the functions we wrote for generating features for the training data expect a list
        of list(s)), and model is a trained LogisticRegression model'''
    n = len(test_sentence)
    T = len(tag_dict)
    Y_pred = np.empty((n-1,T,T))
    Y_start = np.empty((1, T))
    
    index = 0

    for word in test_sentence:
        if index == 0:
            X = build_X([[get_features(index, test_sentence, "<S>", rare_words)]])
            Y_start = model.predict_log_proba(X)
        else:
            for prev_tag in tag_dict.keys():
                j = tag_dict[prev_tag]
                X = build_X([[get_features(index, test_sentence, prev_tag, rare_words)]])
                Y_pred[index-1][j] = model.predict_log_proba(X)
        index += 1
    
    return (Y_pred, Y_start)

def main():
  brown_sentences = brown.tagged_sents(tagset='universal')

  train_sentences = list()
  train_tags = list()
  count = 0

  vocabulary_count = {}
  for sentence in brown_sentences:
    sentence_list = list()
    label_list = list()
    for tags in sentence:
      tags = list(tags)
      ### adding new key to vocabulary counts dictionary, if the key exists add the count to it ######
      vocabulary_count[tags[0]] = vocabulary_count.get(tags[0],0) + 1
     
      sentence_list.append(tags[0])
      label_list.append(tags[1])
   
    train_sentences.append(sentence_list)
    train_tags.append(label_list)
  
  
  for word,count in vocabulary_count.items():   
    if count < 5:
      rare_words.add(word)
            
  ######## training part 3 ####################
  training_features = list()
  
  for train_sentence in train_sentences:
    indx = train_sentences.index(train_sentence)
    word_feature_list = list()
    i = 0
    for word in train_sentence:
      if i == 0:
        prev_tag = '<S>'
      else:
        prev_tag = train_tags[indx][i-1]
      
      word_feature = get_features(i,train_sentence,prev_tag,rare_words)
      word_feature_list.append(word_feature)
      i = i + 1
    training_features.append(word_feature_list)
  
  ### calling the remove rare features #####
  n = 5
  remove_rare_output = remove_rare_features(training_features,n)
  training_features_updated = remove_rare_output[0]
  non_rare_set = remove_rare_output[1]
   
  ################ printing the feature dictionary for the non rare words ##############
  index = 0
  for feature_word in non_rare_set:
    feature_dict[feature_word] = index
    index = index + 1

  tag_vocabulary = set(x for l in train_tags for x in l)
  #### creating tag dictionary where the keys are the 17 tags and the values are the indices assigned to the tag ###
  
  tag_idx = 0
  for tag in tag_vocabulary:
    tag_dict[tag] = tag_idx
    tag_idx = tag_idx + 1
  
  ### calling the build_Y function using the tag_dict #####
  Y_train = build_Y(train_tags)
   
  ##### calling build_X features using the training features #####
  X_train = build_X(training_features_updated)
  model = LogisticRegression(class_weight='balanced',solver='saga',multi_class='multinomial').fit(X_train, Y_train) 

    
  ########## starting part 4 ################
  ### calling the load_test function to read the test file #########
  test_data = load_test('test.txt')

  #### iterating through the test sentences and calling get_predictions for each test_sentence ###########
  ## iterate through each test sentence, using get predictions() and viterbi() to decode the highest-probability sequence of tags
  for test_sentence in test_data:
      y_output = get_predictions(test_sentence,model)
      y_start = y_output[1]
      y_pred = y_output[0]
      predicted_tag_sequence = viterbi(y_start, y_pred)
      print(predicted_tag_sequence)
if __name__ == '__main__':
  main()


