# This code is an example of how to use Gensim's implementation of Word2Vec to train a word embedding model on an Arabic corpus and then use it to get word vectors and find similar words.

# Here's a breakdown of the code:

#     Install necessary libraries: Gensim for Word2Vec, NLTK for tokenization and stop words.
        # pip install gensim
        # pip install nltk
#     Import necessary libraries: Gensim for Word2Vec, NLTK for tokenization and stop words.

#     Download and load Arabic stop words using NLTK.

#     Load an Arabic corpus from a text file.

#     Tokenize the corpus into individual words and convert them to lowercase.

#     Filter out stop words from the tokenized corpus.

#     Train a Word2Vec model on the filtered corpus, specifying a minimum count of 1 for words to be included in the model, a vector size of 100 dimensions for each word vector, and a window size of 5 words.

#     Save the trained model in a file named "arabic_word2vec.model".

#     Get the word vector for a specific word by calling the model's "wv" (word vectors) attribute and passing the word as an argument.

#     Find the most similar words to a specific word by calling the "most_similar()" method on the model's "wv" attribute and passing the word as an argument. This returns a list of tuples, where each tuple contains a similar word and its cosine similarity score.

#     Print the list of similar words.

#     By Mahmoud Abdelhamid :)


from gensim.models import Word2Vec
import nltk
# nltk.download('all')

# download the Arabic stop words from NLTK
nltk.download('stopwords')

# load the Arabic stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('arabic')

# load the Arabic corpus (replace path with your own file path)
with open('arabic_corpus.txt', 'r', encoding='utf-8') as f:
    arabic_corpus = f.read()

# tokenize the corpus
arabic_corpus_tokens = [word.lower() for sent in nltk.sent_tokenize(arabic_corpus) for word in nltk.word_tokenize(sent)]

# filter out stop words
arabic_corpus_tokens = [token for token in arabic_corpus_tokens if token not in stop_words]

# train the Word2Vec model
model = Word2Vec([arabic_corpus_tokens], min_count=1, vector_size=100, window=5)
model.save("arabic_word2vec.model")
# get the word vector for a specific word (replace "الثقافة" with the word you want to get the vector for)
word_vector = model.wv['الثقافة']

# find the most similar words to a specific word (replace "الثقافة" with the word you want to find similar words to)
similar_words = model.wv.most_similar('الثقافة')
print(similar_words)
