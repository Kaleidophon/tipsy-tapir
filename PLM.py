from math import log, exp, cos, pi, sqrt
import numpy as np


class PLM_score:
    # OBS: PLMs need the entire query to score a document. Just the query term_id is not sufficient
    def __init__(self, query_term_ids, document_length, total_number_of_documents, word_positions,
                 background_model, kernel, query_word_positions):
        self.query_term_ids = query_term_ids
        self.document_length = document_length
        self.C = total_number_of_documents
        self.word_positions = word_positions
        self.query_word_positions = query_word_positions
        self.background_model = background_model # A (preferably smoothed) language model of the form back_model[term_id] = counts of the term
        self.kernel = kernel
        self.c_marked_memory = {} # Attempt to make a memory of the values we've already caluclated so we don't have to re-calculate them
        self.c_m_total_memory = dict()
        self.k_cache = dict()
        self.position_indices = list(range(document_length))

    # Static functions
    def c(self, term_id, i):
        # A simple function to check if
        # the word with word id = word_id is at position i in the given document
        return self.word_positions[i] == term_id

    def c_marked(self, term_id, i):
        if (term_id, i) in self.c_marked_memory:
            return self.c_marked_memory[(term_id, i)]
        else:
            c_m = sum([self.c(term_id, i) * self.kernel_func(i, j) for j in self.position_indices])
            self.c_marked_memory[(term_id, i)] = c_m
            return c_m

    def c_m_total(self, i):
        if i not in self.c_m_total_memory:
            self.c_m_total_memory[i] = sum([self.kernel_func(i, j) for j in self.position_indices])
        return self.c_m_total_memory[i]

    def p_w_D_i(self, term_id, i):
        # TODO Refactor
        c_m = self.c_marked(term_id, i)
        # We use the simplification that the sum over all words is simply the sum of the kernel function
        c_m_tot = self.c_m_total(i)
        return c_m / c_m_tot

    def p_w_D_i_smoothed(self, term_id, i, lamb=0.5):  # Dirichlet smoothing
        # TODO Refactor
        Z_i = self.c_m_total(i)
        return (self.c_marked(term_id, i) + lamb * self.background_model[term_id]) / (Z_i + lamb)

    def S(self, i):
        score = 0
        for query_term_id in self.query_term_ids:  # Iterate over the vocabulary of the document
            query_prob = sum([1 for q_id in self.query_term_ids if query_term_id == q_id ]) / len(self.query_term_ids)

            pwdi = self.p_w_D_i(query_term_id, i) \
                if not self.background_model else self.p_w_D_i_smoothed(query_term_id, i)

            if pwdi != 0:
                score += -query_prob * log(query_prob / pwdi)

        return score

    def best_position_strategy_score(self):
        counts = np.zeros(len(self.document_length))
        counts[self.query_word_positions] = 1

        # TODO: Add kernel values
        for position in self.query_word_positions:
            pass


        return max([self.S(i) for i in self.position_indices])

