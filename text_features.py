import pandas as pd
import spacy
import pickle
import numpy as np

nlp = spacy.load("en_core_web_sm")

def spacy_features():
    '''
    Save a dictionary of spacy features into a list of dicts 
    '''
    df = pd.read_csv('image_data.csv')

    # SAVE POSTAGS, SHAPE OF EVERY TOKEN IN ALL CAPTIONS
    spacy_list = []
    for i in range(len(df)):
        if i%1000 == 0:
            print("At location : " + str(i))
        doc = nlp(df.loc[i]["caption"])
        temp_list = []
        temp_list.append(len(doc.ents))
        for token in doc:
            temp_dict = {}
            temp_dict["Text"] = token.text
            temp_dict["POSTag"] = token.pos_
            temp_dict["Shape"] = token.shape_
            temp_dict["IsStart"] = token.is_sent_start
            temp_dict["IsEnd"] = token.is_sent_end
            temp_list.append(temp_dict)
        spacy_list.append(temp_list)

    with open('spacy_features.txt', 'wb') as sf:
        pickle.dump(spacy_list, sf)
    

# spacy_features()
def text_features():
    '''
    Use the saved spacy features to create the dataframe for textual features and merge with image features dataframe
    '''

    with open('spacy_features.txt', 'rb') as sf:
        n_list = pickle.load(sf)
    
    # A DICT OF TEXT FEATURES SUCH AS VERB, NOUN, ADPOSITION, ADJECTIVE, ADVERB, PARTICLE, PROPER NOUN, NUMERAL
    # TAKEN FROM SPACY POSTAG LIST
    # ALSO EXTRACT FEATURES AS TOTAL WORDS, TOTAL ENTITIES, TOTAL CHARACTERS
    text_ft = np.zeros((len(n_list), 21))
    ref_dict = {'VERB':0, 'NOUN':1, 'ADP':2, 'ADJ':3, 'ADV':4, 'PART':5, 'PROPN':6, 'NUM':7,
                'AUX':8, 'CCONJ':9, 'DET':10, 'INTJ':11, 'PRON':12, 'PUNCT':13, 'SCONJ':14,
                'SYM':15, 'X':16, 'SPACE':17, 'ENTS':18, 'CHARS':19, 'WORDS':20}
    
    cols = ['VERB', 'NOUN', 'ADPOSITION', 'ADJECTIVE', 'ADVERB', 'PARTICLE', 'PROPER NOUN', 'NUMBER',
                'AUXILIARY', 'COORDINATING CONJUNCTION', 'DETERMINER', 'INTERJECTION', 'PRONOUN', 'PUNCTUATION',
                'SUBORDINATING CONJUNCTION', 'SYMBOL', 'OTHER', 'SPACE', 'TOTAL ENTITIES', 'CHARACTERS', 'WORDS']
    
    for i in range(len(n_list)):
        # TOTAL ENTITIES
        if i%1000==0:
            print("At index: " + str(i))
        text_ft[i][ref_dict['ENTS']] = n_list[i][0]
        # TOTAL WORDS
        text_ft[i][ref_dict['WORDS']] = len(n_list[i])-1

        # COUNT OF EACH TAG IN A CAPTION
        ch = 0
        for j in range(1, len(n_list[i])):
            curr_dict = n_list[i][j]
            text_ft[i][ref_dict[curr_dict['POSTag']]] += 1
            ch += len(curr_dict['Shape'])

        # TOTAL CHARACTERS
        text_ft[i][ref_dict['CHARS']] += ch

    # SAVE INTO A DATAFRAME
    text_ft = np.array(text_ft, dtype=int)
    df = pd.read_csv('image_data.csv')
    text_df = pd.DataFrame(data=text_ft, columns=cols, index=np.arange(0, len(df)))
    image_text_df = pd.concat([df, text_df], axis=1)
    
    # ALSO EXTRACT TOP 2000 WORDS AND CREATE AN ARRAY USING BAG OF WORDS APPROACH
    classes_df = pd.read_csv('token_frequencies.txt', sep='\t', header=None, names=["Word", "Score"], encoding='latin-1')
    top_k = 2000
    frequent_vocab = []
    for i in range(top_k):
        frequent_vocab.append(classes_df.loc[i]["Word"])

    bow_array = np.zeros((len(df), top_k))

    for i in range(len(df)):
        if i%1000==0:
            print("At location " + str(i))
        capt = df.loc[i]['caption']
        capt = capt.split(' ')
        for k in range(len(capt)):
            for j in range(top_k):
                if capt[k] == frequent_vocab[j]:
                    bow_array[i][j] += 1 

    vocab_cols = []
    for v in range(len(frequent_vocab)):
        word = frequent_vocab[v]
        word = word + " vocab"
        vocab_cols.append(word)

    # print(vocab_cols)
    # SAVE THE FINAL DATAFRAME
    bow_df = pd.DataFrame(data=bow_array, columns=vocab_cols)
    image_text_bow_df = pd.concat([image_text_df, bow_df], axis=1)
    image_text_bow_df.to_csv('image_text_df.csv')

    
# text_features()



