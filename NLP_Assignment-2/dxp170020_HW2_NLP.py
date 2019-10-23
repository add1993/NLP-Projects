import numpy as np
import csv
import re
import nltk
import sys
import sklearn

nltk.download('wordnet')
#np.set_printoptions(threshold=sys.maxsize)
#nltk.download('averaged_perceptron_tagger')
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from nltk import pos_tag
from nltk.corpus import wordnet
from itertools import chain

#length of vocabulary variable initialised to 0
V = 0

### globalvariable counts
counts = {}

reg_both_quotes = re.compile(r"(\'(\S+)\')")
reg_front_quotes =  re.compile(r"(\s+(')(?!(em|70s|tis))(\w+))")
reg_back_quotes = re.compile(r"(\w+)(['])(?!\w)")

#regex for identifying the EDIT_ words
reg_edit_token = re.compile(r"^EDIT_")

#regex for identifying the EDIT_ words while creating dictionary
regex_edit = re.compile(r"EDIT_")

#regex negation
regex_negation = re.compile("^NOT_")

#list to hold excluded words
excluded_words = ('em','tis','70s')

#initializing a set for the negation end tags
negation_end_tags_words = set(['but','however','nonetheless',".","!","?"])
                           
#regex for finding content inside bracket
find_open_bracket = re.compile(r"(\[)(\w+)(\]*)")
find_close_bracket = re.compile(r"(?!\[)(\S+)(\])")
both_brackets = re.compile(r"(\[)(\w+)(\]+)")

##### regex for finding negation words which are listed below ######
reg_negation = re.compile(r"(\b(no|not|never|cannot)\b|(n't))")

#### special case for words like haven't / can't / don't #########

def tag_negation(tokenized_snippet):
  ''' further preprocssing of text'''

  #creating copy of the tokenizes_snippet
  tokenized_s_cpy = tokenized_snippet.copy()

  j = 0
  ### finding the EDIT_ tokens and removing them
  for token in tokenized_s_cpy:
    tokenized_s_cpy[j] = re.sub(reg_edit_token,"",token)
    j = j+1
 

  ###############################################
  #####removing EDIT_ token to process using nltk library and then re adding this token 
  #processing the nltk tags
  nltk_processed_tag = nltk.pos_tag(tokenized_s_cpy)
  
  k = 0
  ### re adding the EDIT_ tags to the processed nltk tags
  for nltk_token in nltk_processed_tag:
    match = reg_edit_token.findall(tokenized_snippet[k])
    if len(match) is 1:
      #tuples are immutable, convert to list before processing it
      list_tuple = ('EDIT_' + nltk_processed_tag[k][0],nltk_processed_tag[k][1])
      nltk_processed_tag[k] = list_tuple
    k = k+1

  
  negate_subsequent = 0
  x = 0

  #iterating through the list of tuples until you find a negation word
  for lib_token in nltk_processed_tag:
    stop_negation = check_end_negation_tag(lib_token)
    negation_match = reg_negation.findall(lib_token[0])
    if stop_negation is 1:
      negate_subsequent = 0
    elif len(negation_match) is 1:
        ### exceptional case for not only!!
      if x < (len(nltk_processed_tag) - 1) and nltk_processed_tag[x+1][0] == "only":
          negate_subsequent = 0
      else:
          ## otherwise if match then add the NOT_token
        negate_subsequent = 1
    elif negate_subsequent is 1:
        list_value = ('NOT_' + nltk_processed_tag[x][0],nltk_processed_tag[x][1])
        nltk_processed_tag[x] = list_value
    x = x+1
    
  return nltk_processed_tag
     
def check_end_negation_tag(in_token):
  #if negation_end_tags_words
  if in_token[0] in negation_end_tags_words:
    return 1
  #check for comparatives
  elif in_token[1] == "JJR" or in_token[1] == "RBR":
    return 1
  else:
    return 0

def load_corpus(corpus_path):
  ''' returns a snippet where snippet is a string and label in an int '''
  final_list = []
  with open(corpus_path, encoding='utf-8', errors='ignore') as file:
    file_reader = csv.reader(file, delimiter='\t')
    for row in file_reader:
      final_list.append(row)
    return final_list
  
def tokenize(snippet):
  ''' takes a string snippet as input and returns a list of tokens'''
  ### check for front quotes, check for back quotes! covers the case 'article , hello' , 'memory'
  m = re.sub(reg_front_quotes, r" \2 \4", snippet)
  n = re.sub(reg_back_quotes, r" \1 \2", m)
  return n.split()

def tag_edits(tokenized_snippet):
  ''' takes as input a list of tokens and returns the same list, 
  but with any square brackets removed and the words inside tagged with EDIT'''
  i = 0
  bracket_count = 0
  for token in tokenized_snippet:
    l = both_brackets.findall(token)
    m = find_open_bracket.findall(token)
    n = find_close_bracket.findall(token)

    if len(l) is not 0:
      tokenized_snippet[i] = re.sub(both_brackets, r"EDIT_\2", token)

    elif len(m) is not 0:
      tokenized_snippet[i] = re.sub(find_open_bracket, r"EDIT_\2", token)
      bracket_count = 1
      
    elif len(n) is not 0:
      tokenized_snippet[i] = re.sub(find_close_bracket, r"EDIT_\1", token)
      bracket_count = 0
      
    elif bracket_count is 1:
      tokenized_snippet[i] = 'EDIT_' + token
    
    i = i + 1

  return tokenized_snippet
    
def get_features(preprocessed_snippet):
    ''' takes the list of tuples from the last preprocessing step, tag negation(), and returns a feature vector in a Numpy array. '''
    numpy_arr = np.zeros(V, dtype = float)
    for word in preprocessed_snippet:
        match_edit = regex_edit.findall(word[0])
        check_word = ""
        found_word = 0

        if len(match_edit) is 1 :
            continue
        else :
            if word[0] in counts:
                found_word = 1
                check_word = word[0]
            else:
                find_negate_reg = regex_negation.findall(word[0])
                if len(find_negate_reg) == 1:
                    orig_word = re.sub(regex_negation,"",word[0])
                    if orig_word in counts:
                        found_word = 1
                        check_word = orig_word

        if found_word is 1:
            indx = list(counts.keys()).index(check_word)
            numpy_arr[indx] = numpy_arr[indx] + 1

    return numpy_arr

def normalize(X):
    ''' takes a feature matrix and normalizes the feature values to be in the range [0, 1] '''
    for col in range(X.shape[1]):
        max_value = np.amax(X[:,col])
        min_value = np.amin(X[:,col])
        
        if max_value > 0 and min_value != max_value:
            X[:,col] = (X[:,col] - min_value)/(max_value - min_value)
    return X

def evaluate_predictions(Y_pred, Y_true):
    ''' takes two Numpy arrays of labels and returns a tuple of floats (precision, recall, fmeasure) '''
    TP = 0
    FP = 0
    FN = 0

    for i in range(len(Y_true)): 
        if Y_true[i]==Y_pred[i]==1:
           TP += 1
        if Y_pred[i]==1 and Y_true[i]!=Y_pred[i]:
           FP += 1
        if Y_pred[i]==0 and Y_true[i]!=Y_pred[i]:
           FN += 1

    return(TP, FP, FN)


def return_dictionary(pass_dict):
    ''' creates a dictionary of unique tokens and their counts as values '''
    i = 0
    for i in pass_dict:
        for vocab in pass_dict[i]:
            match_edit = regex_edit.findall(vocab[0])
            match_negation = regex_negation.findall(vocab[0])
            
            if len(match_edit) is 1 :
                continue
            
            if len(match_negation) is 1:
                original_word = re.sub(regex_negation,"",vocab[0])
                counts[original_word] = counts.get(original_word,0)+1
            
            counts[vocab[0]] = counts.get(vocab[0],0)+1

    return counts

def generate_feature_vector(dictionary, x, y, preprocessed_list):
    ''' self made function which will further call get_feature function '''
    row = 0
    for index in dictionary:
        feature_vector = get_features(dictionary[index])
        x[row] = feature_vector
        y[row] = preprocessed_list[index][1]
        row = row + 1
    return [x,y]


def top_features(logreg_model,k):
    ''' takes a trained LogisticRegression model and an int k and returns a list of length k containing tuples (word, weight) '''
    logistic_regression_coefficient = logreg_model.coef_
    updated_logreg = list()
    to_add = min(logistic_regression_coefficient[0])
    lr_updated = logistic_regression_coefficient[0]
    
    coef_idx = 0
    for coef in lr_updated:
        updated_logreg.append(tuple((coef_idx, coef)))
        coef_idx = coef_idx + 1
    
    ## sorting the coefficients array based on absolute values of weights and in descending order
    sorted_logreg = sorted(updated_logreg, key=lambda x: abs(x[1]),reverse=True)
     
    sorted_idx = 0

    for weight_tuple in sorted_logreg:
        logreg_idx = weight_tuple[0]
        ####### currently for the indices which are larger than the vocabulary, we assign the words as ###########3 
        ################# activeness , evaluation and imagery ###############################################
        if logreg_idx < len(counts):
            vocab = list(counts)[logreg_idx]
            sorted_logreg[sorted_idx] = (vocab,weight_tuple[1])
            sorted_idx = sorted_idx + 1
        elif logreg_idx == len(counts):
            sorted_logreg[sorted_idx] = ("activeness",weight_tuple[1])
        elif logreg_idx == len(counts) + 1:
            sorted_logreg[sorted_idx] = ("evaluation", weight_tuple[1])
        elif logreg_idx == len(counts) + 2:
            sorted_logreg[sorted_idx] = ("imagery",weight_tuple[1])
    
    return sorted_logreg[:k]


####################### Part 4 - Lexicon ##########################################
def load_dal(dal_path):
    ''' Reads in the lexicon in the file dal_path and returns a dictionary where the keys are words, and the values are tuples of floats'''
    dal_list = {}
    with open(dal_path, encoding='utf-8', errors='ignore') as file:
        file_reader = csv.reader(file, delimiter='\t')
        for row in file_reader:
            dal_list[row[0]] = (row[1],row[2],row[3])
    ### deleting the first element since it does not count to the dictionary
    del dal_list['Word']
    return dal_list 

### added a separate method score_snippet_modified for the part 6 of the HW2 #### This is for part 4 of HW2
def score_snippet(preprocessed_snippet, dal):
    ''' takes a fully preprocessed snippet (ie. list of tuples (word, pos)) and the DAL dictionary and returns a tuple of (activeness, pleasantness, imagery) scores for the whole snippet.  '''

    sum_activeness = 0
    sum_evaluation = 0
    sum_imagery = 0

    avg_activeness = 0
    avg_evaluation = 0
    avg_imagery = 0
    
    snippet_count = 1
        
    for sub_snippet in preprocessed_snippet:
        snippet_word  = sub_snippet[0]
        match_not = regex_negation.findall(snippet_word)
        match_edit = regex_edit.findall(snippet_word)
        match_start_edit = reg_edit_token.findall(snippet_word)
        word_scores = list()
        word_activeness = 0
        word_evaluation = 0
        word_imagery = 0

        ######### check if its a NOT_ word with no edits in it##############
        if len(match_not) > 0 and len(match_edit) == 0:
            snippet_word = re.sub(regex_negation,"",snippet_word)
        
        if snippet_word in dal and len(match_start_edit) == 0:
            #### getting the tuple with all three values corresponding to the word #########
            word_scores = dal[snippet_word]
            if len(match_not) > 0:
                word_activeness = -1*float(word_scores[0])
                word_evaluation = -1*float(word_scores[1])
                word_imagery = -1*float(word_scores[2])
            else: 
                word_activeness = float(word_scores[0])
                word_evaluation = float(word_scores[1])
                word_imagery = float(word_scores[2])
        else:
            continue
        
        sum_activeness = sum_activeness + word_activeness
        sum_evaluation = sum_evaluation + word_evaluation
        sum_imagery = sum_imagery + word_imagery
        snippet_count = snippet_count + 1        

    return ((sum_activeness/snippet_count), (sum_evaluation/snippet_count), (sum_imagery/snippet_count))

############## Part 4 - modified get_Features #####################################################################################

def get_features_dal(preprocessed_snippet,dal):
    ''' takes the list of tuples from the last preprocessing step, tag negation(), and returns a feature vector in a Numpy array. '''
    np_arr = np.zeros(V, dtype = float)

    np_arr = get_features(preprocessed_snippet)
    #scores = score_snippet(preprocessed_snippet, dal)

    ############# modified version of score snippet for part 5 ###################
    scores = score_snippet_modified(preprocessed_snippet, dal)
    
    np_arr[V-1] = scores[2]
    np_arr[V-2] = scores[1]
    np_arr[V-3] = scores[0]

    return np_arr

def generate_feature_vector_dal(dictionary, x, dal):
    ''' modified function to handle the part 4 which will call the modified get_features '''
    row = 0
     
    for index in dictionary:
        feature_vector = get_features_dal(dictionary[index],dal)
        x[row] = feature_vector
        row = row + 1
    return x

########### Part 5 #############################################

def get_wordnet_pos(pos_tag):
    """ return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) """
    ## convert to string #####

    str_pos_tag = str(pos_tag)
    if str_pos_tag.startswith('J'):
        return wordnet.ADJ
    elif str_pos_tag.startswith('V'):
        return wordnet.VERB
    elif str_pos_tag.startswith('N'):
        return wordnet.NOUN
    elif str_pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

### modified score snippet for handling the question 6 of HW2 ##########
def score_snippet_modified(preprocessed_snippet, dal):
    ''' takes a fully preprocessed snippet (ie. list of tuples (word, pos)) and the DAL dictionary and returns a tuple of (activeness, pleasantness, imagery) scores for the whole snippet.  '''
    sum_activeness = 0
    sum_evaluation = 0
    sum_imagery = 0

    avg_activeness = 0
    avg_evaluation = 0
    avg_imagery = 0
    
    snippet_count = 1
        
    for sub_snippet in preprocessed_snippet:
        snippet_word  = sub_snippet[0]
        match_not = regex_negation.findall(snippet_word)
        match_edit = regex_edit.findall(snippet_word)
        match_start_edit = reg_edit_token.findall(snippet_word)        
        pos_tag = sub_snippet[1]
        found_word = 0
        found_antonym = 0
        word_scores = list()
        word_activeness = 0
        word_evaluation = 0
        word_imagery = 0

        ######### check if its a NOT_ word with no edits in it##############
        if len(match_not) > 0 and len(match_edit) == 0:
            snippet_word = re.sub(regex_negation,"",snippet_word)
        
        if snippet_word in dal and len(match_start_edit) == 0:
            #### getting the tuple with all three values corresponding to the word #########
            word_scores = dal[snippet_word]
            found_word = 1
        else:
            synonyms = [] 
            antonyms = [] 
            
            wordnet_pos_tag = get_wordnet_pos(pos_tag)

            for syn in wordnet.synsets(snippet_word, pos = wordnet_pos_tag): 
                for l in syn.lemmas(): 
                    synonyms.append(l.name()) 
                    if l.antonyms(): 
                        antonyms.append(l.antonyms()[0].name()) 
                        synonyms = wordnet.synsets(snippet_word,pos = wordnet_pos_tag)
                        lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms])) 
                

            for synonym in set(synonyms):
                if synonym in dal.keys():
                    found_word = 1
                    word_scores = dal[synonym]
                    break;

            ## if word not found in synoym ####

            if found_word is 0:
                for antonym in set(antonyms):
                    if antonym in dal.keys():
                        found_antonym = 1  
                        word_scores = dal[antonym]
                        break;
        
        
        if len(word_scores) > 0:
            ### multiply the scores by -1 if the word found is an antonym or a negation word ##########
            if len(match_not) > 0 or found_antonym == 1:
                word_activeness = -1*float(word_scores[0])
                word_evaluation = -1*float(word_scores[1])
                word_imagery = -1*float(word_scores[2])
            elif found_word == 1: 
                word_activeness = float(word_scores[0])
                word_evaluation = float(word_scores[1])
                word_imagery = float(word_scores[2])
        else :
            continue

        sum_activeness = sum_activeness + word_activeness
        sum_evaluation = sum_evaluation + word_evaluation
        sum_imagery = sum_imagery + word_imagery
        snippet_count = snippet_count + 1

    return ((sum_activeness/snippet_count), (sum_evaluation/snippet_count), (sum_imagery/snippet_count))


if __name__ == '__main__':
    return_list = load_corpus('train2.txt')
    

    ##########creating a dictionary to hold the key-value pair ##########
    vocab_dict = {}
    i = 0
    for sublist in return_list:
      tokenized_snippet = tokenize(sublist[0])
      tokens_with_tags = tag_edits(tokenized_snippet)
      tag_negation_tuple = tag_negation(tokens_with_tags)
      vocab_dict[i] = tag_negation_tuple
      i = i + 1
    
    counts = return_dictionary(vocab_dict)
    V = len(counts)
     
    ######## Unigram Classifier #########################################

    m = len(return_list)
    X_train = np.empty([m, V], dtype = float)
    Y_train = np.empty([m], dtype = int)
    
    ########## generating feature vector for the training example ########
    train_data = generate_feature_vector(vocab_dict, X_train, Y_train, return_list)
    X_train = train_data[0]
    Y_train = train_data[1]

    ## normalizing the X_train ###
    normalized_X = normalize(X_train)
    
    #Create a Gaussian Classifier
    gnb_model = GaussianNB()

    # Train the model using the training sets
    gnb_model.fit(normalized_X,Y_train)

    ###########pre-processing the test data now#########################
    test_dictionary = {}
    
    return_test_list = load_corpus('test.txt')

    test_m = len(return_test_list)
    
    X_test = np.empty([test_m, V], dtype = float)
    Y_true = np.empty([test_m], dtype = int)

    ##########creating a dictionary to hold the key-value pair ##########
    test_dict = {}
    test_index = 0
    for sub_test_list in return_test_list:
      tokenized_test_snippet = tokenize(sub_test_list[0])
      tokens_test_with_tags = tag_edits(tokenized_test_snippet)
      tag_test_negation_tuple = tag_negation(tokens_test_with_tags)
      test_dict[test_index] = tag_test_negation_tuple
      test_index = test_index + 1
    
    
    ####### generating the feature vectors for the test data X_test and Y_true ##############
    test_data = generate_feature_vector(test_dict, X_test, Y_true, return_test_list)
    X_test = test_data[0]
    Y_true = test_data[1]

    normalized_X_test = normalize(X_test)

    Y_predicted = gnb_model.predict(normalized_X_test)

    ############ evaluating the Gaussian NB classifier ###########################
    evaluation = evaluate_predictions(Y_predicted, Y_true)
    tp = evaluation[0]
    fp = evaluation[1]
    fn = evaluation[2]
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f_measure = (2*precision*recall)/(precision + recall)
    print(" precision for the Gaussian NB classifier is ", precision)
    print(" recall for the Gaussian NB Classifier is ", recall)
    print(" fmeasure for the Gaussian NB Classifier is ", f_measure)

    ################ evaluating the Logistic Regression Model ###################
    #Create a LR classifier
    logreg_model = LogisticRegression()
    
    # Train the model using the training sets
    logreg_model.fit(normalized_X,Y_train)

    Y_predicted = logreg_model.predict(normalized_X_test)

    evaluation = evaluate_predictions(Y_predicted, Y_true)
    tp = evaluation[0]
    fp = evaluation[1]
    fn = evaluation[2]
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f_measure = (2*precision*recall)/(precision + recall)
    print(" Precision of Logistic Regression Model is ", precision)
    print(" Recall of the Logistic Regression Model is ", recall)
    print(" F-Mesaure of the Logistic Regression Model is ", f_measure)
    
    ################# calling top_features function #########################################
    k = 10 
    topfeatures = top_features(logreg_model,k)
    print("########### printing the sorted logistic regression with words and weights #########")
    print(topfeatures)    
   
    ############# Lexicon Part 4 ##################################3

    dict_for_sentiments = load_dal("dict_of_affect.txt")
    
    ### increasing the length of vocabulary to V+3 to accomodate three extra weights for activeness, evaluation and imagery ###
    V = len(counts) + 3

    X_train_new = np.empty([m, V], dtype = float)
    
    train_data = generate_feature_vector_dal(vocab_dict, X_train_new, dict_for_sentiments)

    X_train_new = train_data
    normalized_X_train_new = normalize(X_train_new)
    
    X_test_new = np.empty([test_m, V], dtype = float)
    X_test_new = generate_feature_vector_dal(test_dict, X_test_new, dict_for_sentiments)
    normalized_X_test_new = normalize(X_test_new)
    
    ################ evaluating the Logistic Regression Model ###################
    #Create a LR classifier
    
    log_reg_model2 = LogisticRegression()

    # Train the model using the training sets
    log_reg_model2.fit(normalized_X_train_new,Y_train)

    Y_predicted = log_reg_model2.predict(normalized_X_test_new)

    evaluation = evaluate_predictions(Y_predicted, Y_true)
    tp = evaluation[0]
    fp = evaluation[1]
    fn = evaluation[2]
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f_measure = (2*precision*recall)/(precision + recall)
    print(" Precision of Modified Logistic Regression Model ", precision)
    print(" Recall of Modified Logistic Regression Model ",recall)
    print("F-measure of Modified Logistic Regression Model ", f_measure)

    ########## generating top 10 features for this logreg model ######################
    topfeatures = top_features(log_reg_model2,k)
    print("########### printing the sorted logistic regression with words and weights #########")
    print(topfeatures)   


