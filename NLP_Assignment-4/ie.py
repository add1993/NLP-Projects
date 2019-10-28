import nltk
import re
import sys
import csv
from collections import defaultdict 
from nltk import word_tokenize
nltk.download('punkt')
# Fill in the pattern (see Part 2 instructions)
NP_grammar = 'NP: {<DT>?<JJ>*<NN|NNS|NNP|NNPS>+}'


# Fill in the other 4 rules (see Part 3 instructions)
#### hearst patterns with extra credits added ####
hearst_patterns = [ 
    ('((NP_\w+ ?(, )?)+(and |or )?other NP_\w+)', 'after'),
    ('(NP_\w+ (, )?such as (NP_\w+ ? (, )?(and |or )?)+)', 'before'),
    ('(such NP_\w+ (, )?as (NP_\w+ ?(, )?(and |or )?)+)','before'),
    ('(NP_\w+ (, )?including (NP_\w+ ?(, )?(and |or )?)+)', 'before'),
    ('(NP_\w+ (, )?especially (NP_\w+ ?(, )?(and |or )?)+)', 'before'),
    #### extra credit hearst patterns added here ##################
    ('(NP_\w+ (, )?other than (NP_\w+ ? (, )?(and |or )?)+)','before'),
    ('(NP_\\w+ (, )?like (NP_\\w+ ? (, )?(and |or )?)+)','before'),
    ('such (NP_\\w+ (, )?as (NP_\\w+ ? (, )?(and |or )?)+)','before'),
    ('(NP_\\w+ (, )?mainly (NP_\\w+ ? (, )?(and |or )?)+)','before'),
    ('(NP_\\w+ (, )?mostly (NP_\\w+ ? (, )?(and |or )?)+)','before'),
    ('(NP_\\w+ (, )?particularly (NP_\\w+ ? ''(, )?(and |or )?)+)','before'),
    ('(NP_\\w+ (, )?principally (NP_\\w+ ? (, )?(and |or )?)+)','before'),
    ('(NP_\\w+ (, )?in particular (NP_\\w+ ? ''(, )?(and |or )?)+)','before'),
    ('(NP_\\w+ (, )?except (NP_\\w+ ? (, )?(and |or )?)+)','before'),
    ('(NP_\\w+ (, )?other than (NP_\\w+ ? (, )?(and |or )?)+)','before'),
    ('((NP_\\w+ ?(, )?)+(and |or )?which look like NP_\\w+)','after'),
    ('(NP_\\w+ (, )?which be similar to (NP_\\w+ ? ''(, )?(and |or )?)+)','after'),
    ('((NP_\\w+ ?(, )?)+(and |or )? NP_\\w+ type)','after'),
    ('(NP_\\w+ (, )?compare to (NP_\\w+ ? (, )?(and |or )?)+)','before'),
    ('((NP_\\w+ ?(, )?)+(and |or )?as NP_\\w+)','after')]


# Fill in the function (see Part 1 instructions)
# Argument type: path - string
# Return type: list of tuples
def load_corpus(path):
    file_reader = open(path,'r')

    final_list = list()
    dummy_count = 0
    for row in file_reader.readlines():
        row_column = row.split('\t')
        sentence = row_column[0]
        lemmatized = row_column[1]
            
        sentence_list = list()
        for word in sentence.split():
            sentence_list.append(word.strip())
            
        lemmatized_list = list()
        for lemma_word in lemmatized.split():
            lemmatized_list.append(lemma_word.strip())
            
        final_list.append((sentence_list,lemmatized_list))
        dummy_count = dummy_count + 1
        #if dummy_count >= 10000:
        #    break
    file_reader.close()        
    return final_list

# Fill in the function (see Part 1 instructions)
# Argument type: path - string
# Return type: tuple of sets
def load_test(path):
    true_set = set()
    false_set = set()

    with open(path, encoding='utf-8', errors='ignore') as file:
        file_reader = csv.reader(file, delimiter='\t')
        
        for row in file_reader:
            if row[2] == "False":
                false_set.add((row[0],row[1]))
            elif row[2] == "True":
                true_set.add((row[0],row[1]))

    return (true_set,false_set)

# Fill in the function (see Part 2 instructions)
# Argument type: sentence, lemmatized - list of strings; parser - nltk.RegexpParser
# Return type: string
def chunk_lemmatized_sentence(sentence, lemmatized, parser):

    tagged_sentence = nltk.pos_tag(sentence) 
    tagged_lemmatized = list()
    i = 0
    for lemmatize in lemmatized:
        tagged_lemmatized.append((lemmatize,tagged_sentence[i][1]))
        i = i + 1
    nltk_tree = parser.parse(tagged_lemmatized) 
    
    chunks_of_tree = tree_to_chunks(nltk_tree)
    
    merged_chunks = merge_chunks(chunks_of_tree)
    return merged_chunks

# Fill in the function (see Part 2 instructions)
# Helper function for chunk_lemmatized_sentence()
# Argument type: tree - nltk.Tree
# Return type: list of strings
def tree_to_chunks(tree):
    
    current_chunk = list()
    for i in tree:
        if isinstance(i,nltk.Tree):
            current_chunk.append("NP_" + "_".join([token for token, pos in i.leaves()]))
        else:
            current_chunk.append(i[0])

    return current_chunk

# Fill in the function (see Part 2 instructions)
# Helper function for chunk_lemmatized_sentence()
# Argument type: chunks - list of strings
# Return type: string
def merge_chunks(chunks):
    reconstructed_buffer = list()
    for string in chunks:
        if len(reconstructed_buffer) is 0:
            reconstructed_buffer.append(string)
        else:
            if reconstructed_buffer[-1].startswith('NP_') and string.startswith('NP_'):
                updated_string = reconstructed_buffer[-1] + "_" + string[3:]
                reconstructed_buffer[-1] = updated_string
            else:
                reconstructed_buffer.append(string)

    ## join to convert the buffer into a single string with the tokens separated by a single space ##
    joined_string = " ".join([token for token in reconstructed_buffer])

    return joined_string

# Fill in the function (see Part 4 instructions)
# Argument type: chunked_sentence - string
# Yield type: tuple
def extract_relations(chunked_sentence):
    
    for (hearst_pattern, parser) in hearst_patterns:
        #### using re.search() to look for match for every hearst pattern we created ###
        search_obj = re.search(hearst_pattern, chunked_sentence)
        if search_obj:
            ##### using group(0) to look for a match and splitting the match to tokenize ####
            matched_tokens = search_obj.group(0)
            #### removing the remove all tokens from the list that do not have the NP tag. ####
            NPs = [item for item in matched_tokens.split() if item.startswith('NP_')]
            #### calling the postprocess_NPs function ####
            refined_NP_tokens = postprocess_NPs(NPs)
            hypernym = ""
            hyponyms = list()
            if parser == "before":
                hypernym = refined_NP_tokens[0]
                hyponyms = refined_NP_tokens[1:]
            elif parser == "after":
                hypernym = refined_NP_tokens[-1]
                hyponyms = refined_NP_tokens[:-1]
           

            #### iterating through all the hyponym ##########
            for hyponym in hyponyms:
                yield (hyponym, hypernym)


# Fill in the function (see Part 4 instructions)
# Helper function for extract_relations()
# Argument type: list of strings
# Return type: list of strings
def postprocess_NPs(NPs):
    ''' The helper function gets rid of the NP tag and the underscores that we added back in '''
    i = 0
    for NP in NPs:
        updated_string = NP.replace('NP_','')
        updated_string = updated_string.replace('_',' ')
        NPs[i] = updated_string
        i = i + 1

    return NPs

# Fill in the function (see Part 5 instructions)
# Argument type: extractions, gold_true, gold_false - set of tuples
# Return type: tuple
def evaluate_extractions(extractions, gold_true, gold_false):
    ''' The argument extractions is a set of extracted (hyponym, hypernym) tuples, and gold true and gold false are the two gold standard sets from Part 1. The function should return a tuple of (precision, recall, f-measure) '''
    extractions = (gold_true | gold_false) & extractions
    tp = len(extractions & gold_true)
    fp = len(extractions & gold_false)
    fn = len(gold_true - extractions)

    prec = float(tp)/float(tp + fp)
    recall = float(tp)/float(tp + fn)
    f1 = 2*prec*recall/(prec + recall)

    return prec, recall, f1

def main(args):
    corpus_path = args[0]
    test_path = args[1]

    wikipedia_corpus = load_corpus(corpus_path)
    
    test_true, test_false = load_test(test_path)

    NP_chunker = nltk.RegexpParser(NP_grammar)
  
    ######### testing own level #############

    #test_sentence = "such authors as Herrick, Goldsmith, and Shakespeare ."
    #test_sentence = "I am going to get green vegetables such as spinach, peas and kale ."
    #test_sentence = "I like to listen to NP_music from NP_musical_genres such as NP_blues , NP_rock and NP_jazz ."
    
    #test_tokens = word_tokenize(test_sentence)
    #print(test_tokens)
    #test_chunk = chunk_lemmatized_sentence(test_tokens,test_tokens, NP_chunker)
    #print(test_chunk)
    #relations = extract_relations(test_chunk)
    
    #sys.exit()
    ###### testing finish ##################
    

    # Complete the line (see Part 2 instructions)
    wikipedia_corpus = [chunk_lemmatized_sentence(sentence,lemmatized, NP_chunker) for sentence,lemmatized in wikipedia_corpus]
    
    extracted_pairs = set()
    for chunked_sentence in wikipedia_corpus:
        for pair in extract_relations(chunked_sentence):
            extracted_pairs.add(pair)
    print('Precision: %f\nRecall:%f\nF-measure: %f' % evaluate_extractions(extracted_pairs, test_true, test_false))

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
