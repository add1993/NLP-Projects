import re
from collections import Counter
from math import *
import random 
random.seed( 1 )
###################################
#######Author - Divya Porwal######
#######NLP - Assignment 1 #######

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~ ' * n

def end_pad():
    ''' Returns a stop token added to the end of the text '''
    return  ' ~'

def get_tokens(in_str):
    s = in_str.lower()
    # Replace all none alphanumeric characters with spaces
    s = re.sub(r'[^a-zA-Z0-9\s\~]', ' ', s)
    
    # Break sentence in the token, remove empty tokens
    tokens = [token for token in s.split(" ") if token != ""]
    
    return tokens

def context_tuples(tokens,n):
    ''' returns context tuples for a given text '''
    if n is 1 :
        return zip()
    else :
        return zip(*[tokens[i:] for i in range(n-1)])

def generate_ngrams(s, n):

    # Break sentence in the token, remove empty tokens
    tokens = get_tokens(s)
    
    ngrams = context_tuples(tokens,n)
    
    words = [tokens[i+n-1:] for i in range(1)][0]
    if n is 2:
        ngrams = [tokens[i:] for i in range(1)][0]
        merged_list = tuple(zip(words,ngrams))
    elif n is 1:
        merged_list = tuple(tokens)
    else:
        merged_list = tuple(zip(words,list(ngrams))) 

    yield merged_list
    
def create_ngramlm(n, corpus_path,delta):
    ''' returns an NGramLM trained on the data in the file corpus path. '''
    model = NgramLM(n,delta)
    with open(corpus_path, encoding= 'utf-8', errors='ignore') as f:
        for line in f:
            model.update(start_pad(n-1) + line.strip() + end_pad())
    return model

def create_ngramlm_masked(n, corpus_path):
    ''' returns an NGramLM trained on the data in the file corpus path. '''
    mask_rare(corpus_path)
    model = NgramLM(n)
    with open(corpus_path, encoding= 'utf-8', errors='ignore') as f:
        for line in f:
            model.update(start_pad(n-1) + line.strip() + end_pad())
    return model
   
def text_prob(model, text):
    '''  returns the log probablity of the text, under model trained on NgramLM 
        Multiply together its ngram probablities'''
    ngrams_for_text = generate_ngrams(text,model.n)
    prob = 0
    for ngram in list(ngrams_for_text)[0]:
        word = ngram[0]
        if model.n is 1:
            context = ()
        else:
            context = ngram[1]
        ngram_prob = log(model.word_prob(word,context))
        prob = ngram_prob + prob
    return prob

def get_interpolator_text_prob(model_object, text,delta=0):
    ''' return the probablity of the text, using NgramLM interpolator model '''

    ngrams_for_texts = generate_ngrams(text,model_object.ngramLms[n-1].n)
    list_new = list(ngrams_for_texts)[0]
    prob = 0
    for ngram in list_new:
        ngram_prob = model_object.word_prob(ngram[0],ngram[1])
        prob = log(ngram_prob) + prob
        
    return prob

def mask_rare(corpus_path):
    word_list = list()
    internal_vocab = list()
    with open(corpus_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            internal_vocab.extend(get_tokens(line.strip()))
            word_list = Counter(internal_vocab)
    
    with open(corpus_path, 'r') as file :
        filedata = file.read()
        
    for word in word_list:
        if word_list[word] is 1:
            filedata = re.sub(r"\b%s\b" % word , '</unk>', filedata)

    # Write the file out again
    with open(corpus_path, 'w') as file:
        file.write(filedata)

    return file

def create_ngramlm_interpolator(n, corpus_path, lambdas, delta):
    NGramInterpolator_obj = NGramInterpolator(n, lambdas, delta)
    with open(corpus_path,'r') as file:
        for line in file:
            NGramInterpolator_obj.update(start_pad(n-1) + line.strip() + end_pad())
    return NGramInterpolator_obj

class NGramInterpolator:
    def __init__(self, n, lambdas,delta):
        self.n = n
        self.delta = delta
        #lambdas is in descending order of ngram size
        self.lambdas = lambdas
        self.ngramLms = [None]*n
        idx = n-1
        
        #initialise n ngramLM
        while(n > 0):
            self.ngramLms[idx] = NgramLM(n,self.delta)
            idx = idx - 1
            n = n - 1

    def update(self, text):
        for ngramLm in self.ngramLms:
            ngramLm.update(text)
     
    def word_prob(self, word, context):
        probablities = [None]*self.n
        i = 0
        print(word)
        print(context)
        for ngramLm in self.ngramLms:
            probablities[i] = ngramLm.word_prob(word, context)
            context_list = list(context)
            if len(context_list) > 0:
                context_list.pop(0)
            context = tuple(context_list)
            print(context)
            i = i + 1
        j = 0
        for probablity in probablities:
            final_prob = lambdas[j]*probablities[j]

        print(final_prob)
        return final_prob


class NgramLM:
    ''' A basic n-gram model '''

    def __init__(self, n, delta):
        self.n = n
        self.delta = delta
        self.ngram_count = list()
        #initialize vocabulary to 'stop words, start words, unknown' set
        self.vocabulary = set('~')
        #initialize context_counts to 0
        self.context_counts = list()
        self.internal_context_count = list()
        self.internal_ngram_count = list()
        self.internal_vocab = list()
        self.word_list = list()

    def update(self, text):
        ''' Updates the NGramLM’s internal counts and vocabulary for the n-grams in text, 
            which is again a list of words/strings.'''
        if text:
            return_list = generate_ngrams(text,self.n)
            tokens = get_tokens(text)
            context_tup = context_tuples(tokens,n)
            self.internal_context_count.extend(context_tup)
            self.context_counts = Counter(self.internal_context_count)
            self.internal_ngram_count.extend(list(return_list)[0])
            self.ngram_count = Counter(self.internal_ngram_count)
            self.vocabulary = self.vocabulary.union(set(tokens))
            self.internal_vocab.extend(tokens)
            self.word_list = Counter(self.internal_vocab)
                    
    def word_prob(self, word, context):
        ''' returns the probability of the n-gram (word, context) using the model’s internal counters '''
        tuple_list = (word,context)
        if self.context_counts[context] is 0 or self.word_list[word] is 0 or self.ngram_count[tuple_list] is 0 or len(list(context)) is 0:
            #prob =  1/(len(self.vocabulary))
            #prob = 1/self.word_list['unk']
            prob = (self.word_list[word] + self.delta)/(len(list(self.word_list)) + self.delta*len(list(self.word_list)))
        else:
            prob = (self.ngram_count[tuple_list] + self.delta)/(self.context_counts[context] + self.delta*len(list(self.word_list)))
       
        return prob

    def random_word(self, context):
        sorted_list = sorted(self.vocabulary)
        r = random.random()
        sum = 0
        word_list = {}
        previous_sum = 0
        return_word = 0
        for word in sorted_list :
            prob = self.word_prob(word,context)
            sum = sum + prob
            word_list[word] = sum
            if sum >= r and previous_sum < r:
                return_word = word
            previous_sum = sum
        return return_word

    def likeliest_word(self, context):
        ''' returns ngram with the highest probablity for context '''
        sorted_list = sorted(self.vocabulary)
        max_prob = 0
        max_word = ""
        for word in sorted_list:
            if word is not '~':
                prob = self.word_prob(word,context)
                if prob >= max_prob:
                    max_prob = prob
                    max_word = word

        return max_word

def likeliest_text(model, maxlength):
    ''' generates up to max length words, using the previously generated words as context for each new word.
        The initial context should consist of start tokens ‘<s>’, and the function should return the generated 
        string immediately if the stop token ‘</s>’ is generated. '''
    sentences = 5
    list_of_sentences = [None]*sentences
    for sentence in range(sentences):
        previous_context = ''
        initial_context = ('~',) * (model.n - 1)
        for i in range(maxlength):
            new_context = model.likeliest_word(initial_context)
            tuple_list = list(initial_context)
            tuple_list.pop(0)
            orig_tuple = tuple(tuple_list)
            initial_context = orig_tuple + (new_context,)
            previous_context = previous_context + ' ' + new_context
        list_of_sentences[sentence] = previous_context 

    return list_of_sentences


def random_text(model,maxlength):
    ''' generates up to max length words, using the previously generated words as context for each new word.
        The initial context should consist of start tokens ‘<s>’, and the function should return the generated 
        string immediately if the stop token ‘</s>’ is generated. '''

    sentences = 5
    list_of_sentences = [None]*sentences
    previous_context = ''
    for sentence in range(sentences):
        previous_context = ''
        initial_context = ('~',) * (model.n - 1)
        for i in range(maxlength):
            new_context = model.random_word(initial_context)
            tuple_list = list(initial_context)
            tuple_list.pop(0)
            orig_tuple = tuple(tuple_list)
            initial_context = orig_tuple + (new_context,)
            previous_context = previous_context + ' ' + new_context
        list_of_sentences[sentence] = previous_context 
        
    return list_of_sentences


def perplexity(corpus_path, model):
    word_count = list()
    internal_list = list()
    N = 0
    with open(corpus_path,'r') as file:
        for line in file:
            text = line.strip()
            word_list = get_tokens(text)
            internal_list.extend(word_list)
            word_count = Counter(internal_list)
    for word in word_count:
        N = N + word_count[word]

    final_prob = 0
    with open(corpus_path,'r') as file:
        for line in file:
            text = start_pad(model.n-1) + line.strip() + end_pad()
            word_list = get_tokens(text)
            p1 = text_prob(model,text)
            final_prob = final_prob + p1
    final_prob = 1/final_prob
    #convert log to normal value since its negative
    final_prob = pow(exp(final_prob), 1/N)
    return final_prob
    

if __name__ == '__main__':

    ######################################################
    #training_1.txt is the warpeace.txt file
    #train_2.txt is the warpeace.txt file
    #train_3.txt is the shakespeare.txt file
    ################Question  1 ###########################
    n = 3
    corpus_path = 'train_1.txt'
    out_model = create_ngramlm(n,corpus_path,delta=0)
    s1 = "God has given it to me, let him who touches it beware!"
    s2 = "Where is the prince, my Dauphin?"
    p1 = text_prob(out_model,start_pad(n-1) + s1 + end_pad())
    print("###########probablity for the sentence s1 = ", p1)    
    ####erronous statement########
    #p2 = text_prob(out_model,start_pad(n-1) + s2 + end_pad())
    #print("###########probablity for the sentence s1 = ", p2)

    ################# Question 2 #########################
    corpus_path = 'train_2.txt'
    out_model = create_ngramlm_masked(n,corpus_path,delta=0)
    s1 = "God has given it to me, let him who touches it beware!"
    s2 = "Where is the prince, my Dauphin?"
    p1 = text_prob(out_model,start_pad(n-1) + s2 + end_pad())
    print("####probablity for corpus_path p1 #####", p1)
   
    #out_model_delta1 = create_ngramlm(n,corpus_path,delta=1)
    #out_model_delta2 = create_ngramlm(n,corpus_path,delta=.6)
    #out_model_delta3 = create_ngramlm(n,corpus_path,delta=1)
    #out_model_delta4 = create_ngramlm(n,corpus_path,delta=.6)

    #p1 = text_prob(out_model_delta1,start_pad(n-1) + s1 + end_pad())
    #p2 = text_prob(out_model_delta2,start_pad(n-1) + s1 + end_pad())
    #p3 = text_prob(out_model_delta3,start_pad(n-1) + s2 + end_pad())
    #p4 = text_prob(out_model_delta4,start_pad(n-1) + s2 + end_pad())

    #print("##########laplacian probablity for s1 ####", p1)
    #print("######### laplacian probablity for s1 ####", p2)
    #print("##########laplacian probablity for s2 ####", p3)
    #print("##########laplacian probablity for s2 ####", p4)
   
    #n = 3
    #lambdas = [.33,.33,.33]
    #delta = .001
    #corpus_path = 'train_1.txt'
    #out_model_interpolator = create_ngramlm_interpolator(n,corpus_path,lambdas,delta)
    #interpolator_p1 = get_interpolator_text_prob(out_model_interpolator,start_pad(n-1) + s1 + end_pad())
    #interpolator_p2 = get_interpolator_text_prob(out_model_interpolator,start_pad(n-1) + s2 + end_pad())
    #print("###########probablity of interpolator p1 = ", interpolator_p1)
    #print("###########probablity of interpolator p2 = ", interpolator_p2)

    #####################################################################################################

    ###############Question - 3 #############################

    n = 3
    #corpus_path = 'train_3.txt'
    test_path = 'sonnets.txt'
    #out_model_perplexity_1 = create_ngramlm(n,corpus_path,delta=0)
    #out_model_perplexity_2 = create_ngramlm(n,corpus_path,delta=.5)
    #perp1 = perplexity(test_path,out_model_perplexity_1)
    #perp2 = perplexity(test_path,out_model_perplexity_2)
    #print("#################### pperplexity without smoothing########", list(perp1))
    #print("####################perplexity with smoothing ###########", perp2)

    #corpus_path1 = 'train_3.txt'
    #corpus_path2 = 'train_1.txt'
    #model_perp1 = create_ngramlm(n,corpus_path1,delta=.5)
    #model_perp2 = create_ngramlm(n,corpus_path2,delta=.5)
    #perp1 = perplexity(test_path,model_perp1)
    #perp2 = perplexity(test_path,model_perp2)

    #print("############ perplexity for shakespear train data ", perp1)
    #print("############### perplexity for warpeace train data ", perp2)

    #################################################################################

    #########################Question -- 4#############################################
    n = 3
    max_length = 10
    corpus_path = 'train_3.txt'
    #out_model_4 = create_ngramlm(n,corpus_path,delta=.5)
    #print(random_text(out_model_4,max_length))


    ########################4.2#####################
    out_model_4_bigram = create_ngramlm(2,corpus_path,delta=.5)
    out_model_4_trigram = create_ngramlm(3,corpus_path,delta=.5)
    out_model_4_4gram = create_ngramlm(4,corpus_path,delta=.5)
    out_model_4_5gram = create_ngramlm(5,corpus_path,delta=.5)


    #print(likeliest_text(out_model_4_bigram,max_length))
    #print(likeliest_text(out_model_4_trigram,max_length))
    #print(likeliest_text(out_model_4_4gram,max_length))
    #print(likeliest_text(out_model_4_5gram,max_length))



    

