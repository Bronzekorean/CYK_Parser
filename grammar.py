import math
import time


class Gsymbol(object):
    """ implements a grammatical symbol used for cfg
    hash and equality operators are needed to create symbol set and probability map"""

    def __init__(self, symb, isterminal):
        self.symb = symb
        self.isterminal = isterminal

    def __eq__(self, other):
        return self.symb == other.symb and self.isterminal == other.isterminal

    def __lt__(self, other):
        return (self.isterminal, self.symb) < (other.isterminal, other.symb)

    def __hash__(self):
        return hash((self.isterminal, self.symb))

    def __repr__(self):
        return self.symb + ' ' + ('Terminal' if self.isterminal else 'Non_Terminal')


class Grule(object):
    """ implements a grammatical rule used for cfg
        hash and equality operators are needed to create symbol set and probability map"""

    def __init__(self, gsymb, expansion):
        if not isinstance(gsymb, Gsymbol) or gsymb.isterminal:
            raise Exception('Wrong input grammatical symbol')
        else:
            self.gsymb = gsymb
            self.expansion = tuple(expansion)

    def __eq__(self, other):
        if not isinstance(other, Grule):
            raise Exception('Cannot compare grammatical rule to object of type' + str(type(other)))
        else:
            return self.gsymb == other.gsymb and self.expansion == other.expansion

    def __lt__(self, other):
        return (self.gsymb,) + self.expansion < (other.gsymb,) + other.expansion

    def __hash__(self):
        return hash((self.gsymb,) + self.expansion)

    def __repr__(self):
        return self.gsymb.symb + ' -> ' + ', '.join([symbol.symb for symbol in self.expansion])

    def is_chomsky_normal_form(self):
        """
        Used only to make sure the resulting pcfg is in CNF
        
        Returns
            ----------
            bool: True if the rule is in chomsky normal form
        """
        if len(self.expansion) == 1 and self.expansion[0].isterminal:
            return True
        elif len(self.expansion) == 2 and not self.expansion[0].isterminal and \
                not self.expansion[1].isterminal:
            return True
        else:
            return False

    def is_unit_rule(self):
        return len(self.expansion) == 1 and not self.expansion[0].isterminal

    def copy(self):
        return Grule(self.gsymb, self.expansion)


def gsymb_from_string(symb):
    if '(' in symb:
        symb = symb.split('-')[0]  # remove hyphen from NT symbols
        return Gsymbol(symb.replace('(', ''), False)
    else:
        return Gsymbol(symb.replace(')', ''), True)


def level_list(line):
    lvl_list = []
    level = 0
    symbols = line.split(' ')
    for symbol in symbols:
        lvl_list.append(level)
        if '(' in symbol:
            level += 1
        else:
            level -= symbol.count(')')
    return lvl_list


def remove_unit_rule(rule_list):
    """
        remove_nt_to_nt
         A->B, B->C, C->D (D != NT)

         and replace with: A

         Returns:
             new rule_list with no
    """

    new_rule_list = list()
    map_nt_to_nt = {}

    # For each Grule
    for rule in rule_list:
        new_expansion = rule.expansion
        while len(new_expansion) == 1 and not new_expansion[0].isterminal:
            # Dig until find a terminal or more than 2 non-terminals
            for rule2 in rule_list:
                if rule2.gsymb == new_expansion[0]:
                    if rule2.gsymb not in map_nt_to_nt.keys():
                        map_nt_to_nt[rule2.gsymb] = set()
                    map_nt_to_nt[rule2.gsymb].add(rule.gsymb)
                    new_expansion = rule2.expansion
                    break
        new_rule_list.append(Grule(rule.gsymb, new_expansion))

    # Append all missing Grammatical rules
    # Initial length of the list
    n_init = len(new_rule_list)
    for i in range(n_init):
        symbol_to_map = new_rule_list[i].gsymb
        if symbol_to_map in map_nt_to_nt:
            for gsymb in map_nt_to_nt[symbol_to_map]:
                new_rule_list.append(Grule(gsymb, new_rule_list[i].expansion))

    return new_rule_list


def rules_from_line(line):
    tree = dict()
    levels = level_list(line)

    stack = [len(levels) - 1]
    for i in range(len(levels) - 2, -1, -1):
        while stack and levels[stack[-1]] == levels[i] + 1:
            if i not in tree:
                tree[i] = []
            tree[i].append(stack.pop())
        stack.append(i)

    symbols = [gsymb_from_string(symb) for symb in line.split(' ')]

    rule_list = []
    for root, expansion in tree.items():
        root = symbols[root]
        expansion = [symbols[symb] for symb in expansion]
        rule_list.append(Grule(root, expansion))
    return remove_unit_rule(rule_list)


class PCFG(object):

    def __init__(self, corpus, verbose=True, to_lower=True):
        """initialise a new probabilistic CFG from a training corpus
            args
            ------------
            corpus: iterable bracketed parsed sentences
            to_lower: bool if true ignores capitals (useful to decrease lexicon size)
            verbose: bool level of verbosity
        """
        self.verbose = verbose
        self.to_lower = to_lower
        self.count_rules = dict()

        if self.verbose:
            print('reading ', len(corpus), ' lines')
        for line in corpus:

            if to_lower:
                line = line.lower()

            for rule in rules_from_line(line):
                if rule in self.count_rules:
                    self.count_rules[rule] += 1
                else:
                    self.count_rules[rule] = 1

        self.normaliser = None
        self.log_probabilities = None
        self.lexicon = None
        self.terminals = None
        self.reverse_table = None

        self.update_from_counter()

    def __str__(self):
        representation = 'Grammar of size '+ str(len(self.count_rules)) + '\n Rules and log probabilities: \n'
        for rule, log_proba in self.log_probabilities.items():
            representation += str(rule) + ' : ' + str(round(log_proba, 3)) + '\n'
        return representation

    def update_from_counter(self):
        left_hand_side = set([rule.gsymb for rule in self.count_rules.keys()])
        self.normaliser = {symbol: sum(count for rule, count in self.count_rules.items() if rule.gsymb == symbol)
                           for symbol in left_hand_side}

        self.log_probabilities = dict()
        for rule in self.count_rules:
            self.log_probabilities[rule] = math.log(self.count_rules[rule] / float(self.normaliser[rule.gsymb]))

        self.lexicon = [[rule.gsymb] + list(rule.expansion) for rule in self.count_rules]
        self.lexicon = set([symbol for rule in self.lexicon for symbol in rule])
        self.terminals = set([gsymb for gsymb in self.lexicon if gsymb.isterminal])

        # reverse the rules to be able to run CYK algorithm
        self.reverse_table = dict()
        for rule, log_proba in self.log_probabilities.items():
            if rule.expansion in self.reverse_table:
                self.reverse_table[rule.expansion].append((rule, log_proba))
            else:
                self.reverse_table[rule.expansion] = [(rule, log_proba)]

    def chomsky_transform(self):
        """
            Transforms the grammar into Chomsky Normal Form and keeping the same probabilities

            We assume the root symbol never appears on the Right Hand Side

            Returns
            ----------
            None
        """

        if self.verbose:
            print('transforming grammar of size', len(self.count_rules))
            print('Eliminate rules with non-solitary terminals and merging.')
            start = time.time()

        count_rules_copy = set(self.count_rules.keys()).copy()
        # replace the internal T with NT
        for rule in count_rules_copy:
            if len(rule.expansion) > 1:
                # create copy of the rule to modify
                modified_rule = Grule(rule.gsymb, rule.expansion)

                for i, gsymb in enumerate(rule.expansion):

                    if gsymb.isterminal:
                        new_symb = Gsymbol("NT_" + gsymb.symb, False)
                        new_rule = Grule(new_symb, [gsymb])
                        self.count_rules[new_rule] = 1

                        # Replace old terminal by new non-terminal
                        new_expansion = list(modified_rule.expansion)
                        new_expansion[i] = new_symb
                        modified_rule.expansion = new_expansion

                index = len(rule.expansion) - 1
                new_count = self.count_rules.pop(rule)
                while index > 1:
                    # New key is the concatenation of the two keys
                    new_key = modified_rule.expansion[index - 1].symb + "_" + modified_rule.expansion[index].symb
                    new_symb = Gsymbol(new_key, False)

                    # Make new transition
                    new_rule = Grule(new_symb, [modified_rule.expansion[index - 1], modified_rule.expansion[index]])

                    new_expansion = list(modified_rule.expansion)
                    new_expansion = new_expansion[:-2]
                    new_expansion.append(new_symb)
                    modified_rule.expansion = tuple(new_expansion)

                    if new_rule in self.count_rules:
                        self.count_rules[new_rule] += new_count
                    else:
                        self.count_rules[new_rule] = new_count
                    index -= 1
                self.count_rules[modified_rule] = new_count

        if self.verbose:
            print('Updating probabilities')

        self.update_from_counter()

        if self.verbose:
            print('Grammar transformed into CNF took ', time.time() - start, ' s')
            print('New Grammar size ',  len(self.count_rules))
