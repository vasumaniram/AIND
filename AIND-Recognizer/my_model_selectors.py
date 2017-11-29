import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        log_model_list = []
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                                  random_state=self.random_state, verbose=False).fit(self.X,
                                                                                                     self.lengths)
                logL = hmm_model.score(self.X,self.lengths)
                N = len(self.sequences)
                f = len(self.X[0])
                p = num_states ** 2 + ( 2 * num_states * f ) - 1
                BIC = -2 * logL + p * np.log(N)
                log_model_list.append((BIC,hmm_model))
            except ValueError:
                pass

        if len(log_model_list) == 0:
            return None
        else:
            best_model_idx = np.argmin([log_model[0] for log_model in log_model_list])
            return log_model_list[best_model_idx][1]



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        dic_model_list = []
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                                  random_state=self.random_state, verbose=False).fit(self.X,
                                                                                                     self.lengths)
                logL = hmm_model.score(self.X,self.lengths)
                M = len(self.hwords)
                other_scores = []
                for word in self.hwords.keys():
                    if self.this_word != word:
                        other_X,other_lengths = self.hwords[word]
                        other_scores.append(hmm_model.score(other_X,other_lengths))
                other_scores_sum = np.sum(other_scores)
                DIC = logL - ( 1 / ( M - 1 ) ) * other_scores_sum
                dic_model_list.append((DIC,hmm_model))
            except ValueError:
                pass
        if len(dic_model_list) == 0:
            return None
        else:
            best_model_idx = np.argmax([dic_model[0] for dic_model in dic_model_list])
            return dic_model_list[best_model_idx][1]


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection using CV
        log_model_list=[]
        for num_states in range(self.min_n_components,self.max_n_components + 1):
            logL = []
            #print(len(self.sequences))
            try:
                best_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                         random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                split_method = KFold()
                for train_idx,test_idx in split_method.split(self.sequences):
                    train_X,train_lengths = combine_sequences(train_idx,self.sequences)
                    kfold_hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(train_X, train_lengths)
                    test_X,test_lengths = combine_sequences(test_idx,self.sequences)
                    score = kfold_hmm_model.score(test_X,test_lengths)
                    logL.append(score)
                    if num_states == self.min_n_components:
                        best_score = score
                    elif score > best_score:
                        best_score = score
                        best_model = kfold_hmm_model
                log_model_list.append((np.mean(logL), best_model))
            except ValueError:
                pass
        #print(log_model_list)
        if len(log_model_list) == 0:
            return best_model
        else:
            best_model_idx = np.argmax([log_model[0] for log_model in log_model_list])
            return log_model_list[best_model_idx][1]

