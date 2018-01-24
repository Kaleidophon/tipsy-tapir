from math import log, exp, cos, pi, sqrt

class PLM_score():
    # OBS: PLMs need the entire query to score a document. Just the query term_id is not sufficient
    # Static functions
    def c(self, term_id, i):
        # A simple function to check if
        # the word with word id = word_id is at position i in the given document
        return self.word_positions[i] == term_id

    def k_passage(self, i, j):
        return 1 if abs(i - j) <= self.rho else 0

    def k_gaussian(self, i, j):
        return exp(-((i-j)**2) / (2 * (self.rho**2)))

    def k_triangle(self, i, j):
        return 1 - (abs(i - j) / self.rho) if abs(i - j) <= self.rho else 0

    def k_triangle(self, i, j):
        return 0.5 * (1 + cos((abs(i - j) * pi)/self.rho)) if abs(i - j) <= self.rho else 0

    def k_circle(self, i, j):
        return sqrt(1 - ((abs(i - j) / self.rho))**2) if abs(i - j) <= self.rho else 0

    def __init__(self, query_term_ids, document_length, total_number_of_documents, word_positions, query_model, background_model, rho=50):
        self.query_term_ids = query_term_ids
        self.document_length = document_length
        self.C = total_number_of_documents
        self.word_positions = word_positions
        self.query_model = query_model # A model of the form query_model[term_id] = probability of term_id given all queries
        self.background_model = background_model # A (preferably smoothed) language model of the form back_model[term_id] = counts of the term
        self.rho = rho
        self.kernel_func = self.k_gaussian # Set the desired kernel function

    def c_marked(self, term_id, i):
        return sum([self.c(term_id, i) * self.kernel_func(i, j, self.rho) for j in range(self.document_length)])

    def p_w_D_i(self, term_id, i):
        c_m = self.c_marked(term_id, i)
        # We use the simplification that the sum over all words is simply the sum of the kernel function
        c_m_total = sum([self.kernel_func(i, j) for j in range(self.document_length)])
        return c_m / c_m_total

    def S(self, i):
        score = 0
        for term_id in set(self.word_positions): # Iterate over the vocabulary of the document
            if term_id != 0:
                query_prob = self.query_model[term_id]
                if query_prob == 0:
                    continue
                score += -query_prob * log(query_prob/ (self.background_model[term_id] / self.C))

        return score

    def best_position_strategy_score(self):
        return max([self.S(i) for i in range(self.document_length)])

