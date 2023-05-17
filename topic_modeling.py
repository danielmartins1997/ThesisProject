import gensim
import gensim.corpora as corpora
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric

# Functions needed to use the function to remove Topic Features (Topic Modeling) 
# function to convert sentences to words
def sentencesToWords(sentences): 
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True)) # deacc=True -> meaning removes punctuations

# make bigrams
def make_bigrams(texts, dataInput):  
    data = dataInput
    data_words = list(sentencesToWords(data))
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in texts]

# lemmatization, converting a word to its root word. ‘walking’ –> ‘walk’
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# words that represent a topic, can choose in PreProcessing if that words will be ignored
def TopicFeaturesToRemove(normalizedDF, numberFeatures, useTopicFeature):
  if useTopicFeature == "True":  # if it's true ...
        df_initial = normalizedDF
        data = df_initial['utterance'].values.tolist()
        
        # convert sentences to words
        data_words = list(sentencesToWords(data))

        # form bigrams
        data_words_bigrams = make_bigrams(data_words, data)

        # do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=[
            'NOUN', 'ADJ', 'VERB', 'ADV'])

        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)

        # Create Corpus
        texts = data_lemmatized

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=10,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)

        lda_topics = lda_model.show_topics(num_words=numberFeatures)
        
        topics = []
        filters = [lambda x: x.lower(), strip_punctuation, strip_numeric]
        for topic in lda_topics:
            #print(topic)
            topics.append(preprocess_string(topic[1], filters))
            flat_list = [item for sublist in topics for item in sublist]
            #print(flat_list)
            return flat_list
  else:
        flat_list = []
        #print(flat_list)
        return flat_list