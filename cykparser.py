from grammar import Gsymbol, PCFG
from tqdm import tqdm
import pickle
from outofvocabulary import OOVhandler
import time

class CykParser(object):

    def __init__(self, pcfg, root_symbol, handleoov=False, embeddings_path=''):
        """ implements a a CYK parser from a probabilistic grammar, and a root symbol.
            We add the possibility to add an out of the vocabulary handler.
        """
        self.root_symbol = root_symbol
        self.pcfg = pcfg
        self.pcfg.chomsky_transform()
        if handleoov:
            words, embeddings = pickle.load(open(embeddings_path, 'rb'), encoding='latin1')
            self.oofHandler = OOVhandler(pcfg, words, embeddings)
        else:
            self.oofHandler = handleoov

    def bracketed_from_cyk(self, cyk_table, length, k, gsymb, original_sentence):
        """ Returns string representation of a parsed sentence
            if the symbol used is a dummy symbol it will not be printed"""
        cell = cyk_table[length][k]

        rule, _, length1, start1, length2, start2 = cell[gsymb]
        if '_' not in gsymb.symb:
            # checks that this is not a dummy symbol
            bracketed_expression = ' (' + gsymb.symb

            if start2 == length2 == 0:
                return bracketed_expression + ' ' + original_sentence[start1] + ')'
            else:
                return bracketed_expression + self.bracketed_from_cyk(cyk_table, length1, start1, rule.expansion[0]) + \
                         self.bracketed_from_cyk(cyk_table, length2, start2, rule.expansion[1]) + ')'
        else:
            if start2 == length2 == 0:
                return original_sentence[start1]
            else:
                return self.bracketed_from_cyk(cyk_table, length1, start1, rule.expansion[0]) +\
                       self.bracketed_from_cyk(cyk_table, length2, start2, rule.expansion[1])

    def parse_sentence(self, sentence):
        """ Returns a parsed expression of the input sentence
            If the sentence can not be parsed using the grammar returns None."""

        # create tokens from the sentence
        if self.pcfg.to_lower:
            sentence = sentence.lower()

        if self.oofHandler:
            sentence_to_parse = [self.oofHandler.replace(w) for w in sentence.split(' ')]
        else:
            sentence_to_parse = sentence.split(' ')

        sentence_to_parse = [Gsymbol(symb, True) for symb in sentence_to_parse]
        real_sentence = sentence.split(' ')
        n = len(sentence_to_parse)
        cyk_table = [[{} for _ in range(n - i)] for i in range(n)]

        # initialize the first row of the cyk table
        for counter, token in enumerate(sentence_to_parse):
            reversed_table = self.pcfg.reverse_table.get((token,), None)
            if reversed_table:
                for rule, log_proba in reversed_table:
                    cyk_table[0][counter][rule.gsymb] = (rule, log_proba, 0, counter, 0, 0)
            else:
                print('Could not parse sentence due to presence of unknown token :', token)
                return None

        # Construct CYK table
        for length in range(1, n):
            for k in range(n - length):
                # On length,k column, get the possible combinations to get a suitable substring
                # we assume that no symbol can be split more than once (the grammar is in CNF)
                for i in range(length):
                    # Get all symbols on cell (i, k) and cell (lev-i-1, k+1+i)
                    candidate_symbols_1 = cyk_table[i][k].keys()
                    candidate_symbols_2 = cyk_table[length - i - 1][k + 1 + i].keys()
                    for c1 in candidate_symbols_1:
                        for c2 in candidate_symbols_2:

                            likelihood = cyk_table[i][k][c1][1] + cyk_table[length - i - 1][k + 1 + i][c2][1]
                            # Look for transitions
                            for rule, log_proba in self.pcfg.reverse_table.get((c1, c2), {}):
                                # Add combination or replace if probability is greater
                                if rule.gsymb not in cyk_table[length][k].keys() or \
                                        likelihood + log_proba > cyk_table[length][k][rule.gsymb][1]:
                                    cyk_table[length][k][rule.gsymb] = (rule, likelihood + log_proba, i, k,
                                                                        length - i - 1, k + i + 1)
        # Flag used to debug prints the CYK table
        debug = False
        if self.root_symbol not in cyk_table[n - 1][0]:
            if debug:
                print("Unweighted CYK table:")
                for lev in range(n - 1, -1, -1):
                    print([list(symbset.keys()) for symbset in cyk_table[lev]])
                print([gsymb.symb for gsymb in sentence_to_parse])
            return None

        else:
            return "( " + self.bracketed_from_cyk(cyk_table, n - 1, 0, self.root_symbol, real_sentence) + ")"

    def parse(self, sentences):
        """Parses a list of sentences"""

        if self.pcfg.verbose:
            print('parsing ', len(sentences), 'sentences')
        parsed = list()
        failed = 0
        time_list = list()
        for sentence in tqdm(sentences):
            start = time.time()
            parsed_expression = self.parse_sentence(sentence)
            if not parsed_expression:
                parsed_expression = ''
                failed += 1
            end = time.time()
            time_list.append(round(end - start, 2))
            parsed.append(parsed_expression)

        if self.pcfg.verbose:
            print('parsed ', len(sentences) - failed, ', ', failed, 'could not parsed using CYK')

        return parsed, time_list


def readfile(path_file):
    """reads a text file sequentially"""
    with open(path_file, 'r', encoding='utf8') as file:
        lines = file.readlines()
    return lines


def raw_sentences(bracketed_sentences):
    """ returns the raw sentence from a bracketed expression"""
    raws = list()
    for sentence in bracketed_sentences:
        words = sentence.split(' ')
        words = [word.replace(')', '') for word in words if ')' in word]  # keep terminals only
        raw_sentence = ' '.join(words)
        raws.append(raw_sentence)
    return raws


def writefile(pathfile, lines):
    """writes a list of sentences to the pathfile"""
    lines = '\n'.join(lines)
    with open(pathfile, 'w', encoding='utf8') as f:
        f.write(lines)


if __name__ == "__main__":

    training_path = 'training_corpus'
    test_path = 'test_corpus'
    prediction_path = 'prediction'
    result_path = 'result.txt'

    corpus = readfile('sequoia-corpus+fct.mrg_strict')
    corpus = [line[: -1].lower() for line in corpus]
    corpus_size = len(corpus)

    training_corpus = corpus[:int(corpus_size * 0.8)]
    writefile(training_path, training_corpus)

    test_corpus = corpus[int(corpus_size * 0.9):]
    writefile(test_path, test_corpus)
    test_corpus = [line[2: -1].lower() for line in test_corpus]


    # retreiving raw sentences from test
    raw_test = raw_sentences(test_corpus)

    #creating PCFG and running parser
    root = Gsymbol('sent', False)
    learned_pcfg = PCFG(training_corpus, to_lower=True)
    parser = CykParser(learned_pcfg, root, handleoov=True, embeddings_path='polyglot-fr.pkl')

    parsed, time_list = parser.parse(raw_test)
    writefile(prediction_path, parsed)

    import matplotlib.pyplot as plt

    complexity = list()
    sentence_lengths = list(set([len(sentence.split(' ')) for sentence in raw_test]))
    for n in sentence_lengths:
        complexity.append(max(t for length, t in zip([len(sentence.split(' ')) for sentence in raw_test], time_list)
                              if length == n))
    plt.plot(sentence_lengths, complexity)
    plt.title("Time as a function of sentence length")
    plt.xlabel("Sentence length")
    plt.ylabel("Time(s)")
    plt.savefig('complexity.pdf')


