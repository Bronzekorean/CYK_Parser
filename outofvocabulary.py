from operator import itemgetter
import re
import numpy as np


def levenshtein_distance(s, t):
    """
        computes the Levenshtein distance between the strings
        s and t using dynamic programming

        Returns
        ----------
        dist(int): the Levenshtein distance between s and t
    """
    rows = len(s) + 1
    cols = len(t) + 1

    # create matrix and initialise first line and column
    dist = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(1, rows):
        dist[i][0] = i
    for i in range(1, cols):
        dist[0][i] = i

    # use the recursion relation
    # lev(a[:i], lev[:b]) = min(lev(a[:i - 1], b[j]) + 1,lev(a[:i], b[j -1]) + 1,
    #     lev(a[:i - 1], b[j -1]) + ai≠bj)

    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(dist[row - 1][col] + 1,  # deletion
                                 dist[row][col - 1] + 1,  # insertion
                                 dist[row - 1][col - 1] + cost)  # substitution
    return dist[rows - 1][cols - 1]


class OOVhandler(object):

    def __init__(self, pcfg, words, embeddings):
        """ initialize out of vocabulary handler
            vocabulary: the vocabulary of the corpus
            words: words used from a bigger corpus(provided by polyglot)
            embeddings: the vector representation of words"""

        print('creating out of vocabulary handler:')
        self.words = words
        self.word_id = {word: i for i, word in enumerate(self.words)}
        self.embeddings = embeddings

        self.terminals = [terminal.symb for terminal in pcfg.terminals]

        print('Keeping only common words that have embeddings')
        self.embedded_terminals = [terminal for terminal in self.terminals if terminal in words]

    def closer_levenshtein(self, word):
        """
        returns the closest word in the word embedding using the levenshtein distance
        """
        word_distances = [(w, levenshtein_distance(word, w)) for w in self.words]
        return min(word_distances, key=itemgetter(1))[0]

    def case_normalizer(self, word):
        """ In case the word is not available in the vocabulary,
        we can try multiple case normalizing procedure.
        We consider the best substitute to be the one with the lowest index,
        which is equivalent to the most frequent alternative."""
        w = word
        lower = (self.word_id.get(w.lower(), 1e12), w.lower())
        upper = (self.word_id.get(w.upper(), 1e12), w.upper())
        title = (self.word_id.get(w.title(), 1e12), w.title())
        results = [lower, upper, title]
        results.sort()
        index, w = results[0]
        if index != 1e12:
            return w
        return word

    def normalize(self, word):
        """ Find the closest alternative in case the word is OOV."""
        digits = re.compile("[0-9]", re.UNICODE)
        if word not in self.words:
            word = digits.sub("#", word)
        # if the word is not in the vocabulary try different normalizations
        if word not in self.words:
            word = self.case_normalizer(word)

        # if the word is still not in the vocabulary replace it by the closest word
        # using the levenshtein distance
        if word not in self.words:
            return self.closer_levenshtein(word)

        return word

    def nearest_cosine(self, word):
        """ Sorts words according to their Euclidean distance.
            To use cosine distance, embeddings has to be normalized so that their l2 norm is 1.
            Returns
            ----------
            word: closest word in the embedded terminals to the word in input
        """

        e = self.embeddings[self.word_id[word]]
        # normalise e and the embedding matrix
        e = e / np.linalg.norm(e)
        transformed_embeddings = np.array([self.embeddings[self.word_id[w]] for w in self.embedded_terminals])
        transformed_embeddings = transformed_embeddings.T / (np.sum(transformed_embeddings ** 2, axis=1) ** 0.5)
        distances = e @ transformed_embeddings
        return self.embedded_terminals[max(enumerate(distances), key=itemgetter(1))[0]]

    def replace(self, oov_word):
        """Replace an out of the vocabulary word with another terminal word
        Returns
        ----------
        word: string.
            most similar word in the terminal embedded words
        """

        if oov_word in self.terminals:
            return oov_word
        else:
            # first find the closest word in the vocabulary using levenshtein distance
            word = self.normalize(oov_word)
            # find the closest terminal using the cosine similarity.
            return self.nearest_cosine(word)
