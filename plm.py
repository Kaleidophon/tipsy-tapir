from math import log, exp, cos, pi, sqrt
import numpy as np


class PLM:
    # OBS: PLMs need the entire query to score a document. Just the query term_id is not sufficient
    def __init__(self, query_term_ids, document_length, query_term_positions, background_model, kernel, collection_length):
        self.query_term_ids = query_term_ids
        self.document_length = document_length
        self.query_term_positions = query_term_positions
        self.background_model = background_model # A (preferably smoothed) language model of the form back_model[term_id] = counts of the term
        self.collection_length = collection_length # A (preferably smoothed) language model of the form back_model[term_id] = counts of the term
        self.kernel = kernel()
        self.kernel_len = len(self.kernel)
        self.sigma = (len(self.kernel) - 1) // 2
        self.c_marked_memory = {} # Attempt to make a memory of the values we've already caluclated so we don't have to re-calculate them
        self.c_m_total_memory = dict()
        self.k_cache = dict()
        self.position_indices = list(range(document_length))
        self.full_cm_total = None

    def c_m_total(self, i):
        """
        Compute the normalization constant Z_i.
        """
        # Here we're going to apply the mathematical trick from (Lv et al., 2009) section 3.3 where they show that
        # the computation of the normalization factor Z_i can be simplified by realizing that "the sum of propagated
        # counts to a position is equal to that from the position".
        # Therefore we just sum our pre-computed kernel values, taking in account positions that are close to the
        # edges of the document.

        start_kernel = 0 if i >= self.sigma else self.sigma - i  # Use full kernel window or cut it off on the left
        # Use full kernel window or cut it off on the right
        end_kernel = self.kernel_len if i < self.document_length - self.sigma \
                else self.sigma + self.document_length - i  # End of kernel value window

        # Cache full kernel window sum to save time
        if start_kernel == 0 and end_kernel == self.kernel_len:
            if self.full_cm_total is None:
                self.full_cm_total = np.sum(self.kernel)
            return self.full_cm_total

        return np.sum(self.kernel[start_kernel:end_kernel])

    def p_w_D_i(self, i, counts, *args):
        """ Compute the probability of a word at a certain position in a document. """
        c_m = counts[i]
        # We use the simplification that the sum over all words is simply the sum of the kernel function
        c_m_tot = self.c_m_total(i)

        return c_m / c_m_tot

    def p_w_D_i_smoothed(self, i, counts, term_id, mu=1500):  # Dirichlet smoothing
        c_m = counts[i]
        Z_i = self.c_m_total(i)
        return (c_m + mu * (self.background_model[term_id] / self.collection_length)) / (Z_i + mu)

    def propagate_counts(self, query_term_positions):
        """ Propagate the counts of query term to neighboring position using a pre-defined kernel function. """
        counts = np.zeros(self.document_length)
        #counts[query_term_positions] = 1  # Set counts of query words at positions in document to 1

        for position in query_term_positions:
            # Use full kernel window or cut it off on the left
            start_kernel = 0 if position >= self.sigma else self.sigma - position  # Start of kernel value window
            # Use full kernel window or cut it off on the right
            end_kernel = self.kernel_len if position < self.document_length - self.sigma \
                else self.sigma + self.document_length - position  # End of kernel value window

            # Spread values until start of the document or position minus kernel spread
            start_document = max(0, position - self.sigma)  # Start of propagated counts in document
            # Spread values until end of the document or position plus kernel spread
            end_document = min(position + self.sigma + 1, self.document_length)  # End of propagated counts in document

            # Finally add up the propagated counts relative to the position
            counts[start_document:end_document] += self.kernel[start_kernel:end_kernel]

        return counts

    def S(self, i, counts):
        score = -9999
        # Only iterate over the query words, not over all the words in the vocabulary for the following reason:
        # We're looking for the maximum score. p(w|Q) is a ML estimate + relevance feedback for a word given a query.
        # Because we only have the ML estimate, p(w|Q) and therefore the whole score will be 0 for words that don't
        # occur in the query. Given our query is at least of size one, these words cannot provide the max score.
        for query_term_id in self.query_term_ids:
            # p(w|Q) ML estimate
            query_prob = np.sum(np.array([1 for q_id in self.query_term_ids if query_term_id == q_id]))\
                         / len(self.query_term_ids)

            pwdi = self.p_w_D_i(i, counts, query_term_id) \
                if not self.background_model else self.p_w_D_i_smoothed(i, counts, query_term_id)

            if pwdi != 0:
                score += query_prob * log(query_prob / pwdi)

        return -score

    def best_position_strategy_score(self, use_surrounding_positions=False):
        """ Use the best position strategy in order to score a document given a query. """
        counts = self.propagate_counts(self.query_term_positions)

        # Compromise: Consider all the words that lie within the sigma-spread of the kernel function of a query term
        if use_surrounding_positions:
            touched_positions = set()
            for query_term_position in self.query_term_positions:
                for i in range(-self.sigma, self.sigma):
                    pos = query_term_position + i
                    if pos > -1 and pos < self.document_length:
                        touched_positions.add(pos)

            scores = np.array([self.S(i, counts) for i in touched_positions])

        # Strong simplification assumption: Only consider positions of query terms. Original authors of this model
        # admit that this doesn't influence performance significantly (see report).
        else:
            scores = np.array([self.S(i, counts) for i in self.query_term_positions])

        return np.max(scores) if len(scores) > 0 else -9999

