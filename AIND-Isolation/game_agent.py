"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    no_of_spaces = game.get_blank_spaces()
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    is_starting =  len(no_of_spaces) > ( ( game.width * game.height ) - 10 )
    is_end      = len(own_moves) < 5 or len(opp_moves) < 5
    if is_end:
        return float(len(own_moves) - len(opp_moves))
    elif is_starting:
        # corner_moves = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
        #                 (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
        #                 (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),
        #                 (1, 6), (2, 6), (3, 6), (4, 6), (5, 6)]
        # mid_moves = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
        #              (2, 1), (3, 1), (4, 1),
        #              (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),
        #              (2, 5), (3, 5), (4, 5)]
        #
        # centre_moves = [(2, 2), (2, 3), (2, 4),
        #                 (3, 2), (3, 4),
        #                 (4, 2), (4, 3), (4, 4)]
        # own_moves = game.get_legal_moves(player)
        # opp_moves = game.get_legal_moves(player)
        #
        # center_own_moves = 0
        # for m in own_moves:
        #     center_own_moves += 2 if m == (3, 3) else 0
        #     center_own_moves += 1 if m in centre_moves else 0
        #     center_own_moves += 0.5 if m in mid_moves else 0
        #     center_own_moves += 0.25 if m in corner_moves else 0
        # center_opp_moves = 0
        # for m in opp_moves:
        #     center_opp_moves += 2 if m == (3, 3) else 0
        #     center_opp_moves += 1 if m in centre_moves else 0
        #     center_opp_moves += 0.5 if m in mid_moves else 0
        #     center_opp_moves += 0.25 if m in corner_moves else 0
        # 0return float(center_own_moves -  center_opp_moves) #float((len(own_moves) + center_own_moves)  - (len(opp_moves) + center_opp_moves))
        return float( len(own_moves) - (2 * len(opp_moves)))
    else:
        own_next_moves = []
        for m in own_moves:
            own_next_moves.append(len(game.forecast_move(m).get_legal_moves()))
        opp_next_moves = []
        for m in opp_moves:
            opp_next_moves.append(len(game.forecast_move(m).get_legal_moves()))
        return float(
            max(own_next_moves) if bool(own_next_moves) else 0 - max(opp_next_moves) if bool(opp_next_moves) else 0)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - (2 * opp_moves))


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - (3 * opp_moves))


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=20.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    @staticmethod
    def get_best_move(game):
        legal_moves = game.get_legal_moves()
        return legal_moves[0] if bool(legal_moves) else (-1, -1)


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """
    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = MinimaxPlayer.get_best_move(game)
        #best_move = self.minimax(game, self.search_depth)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            best_move = self.minimax(game, self.search_depth)
        except SearchTimeout:
            return best_move  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # TODO: finish this function!
        best_move = MinimaxPlayer.get_best_move(game)
        if bool(game.get_legal_moves()):
            best_move = max(game.get_legal_moves(), key=lambda x: self.min_value(game.forecast_move(x), depth-1))
        return best_move

    @staticmethod
    def terminal_test(self, gameState):
        """ Return True if the game is over for the active player
        and False otherwise.
        """
        # if self.time_left() < self.TIMER_THRESHOLD:
        #     raise SearchTimeout()
        return not bool(gameState.get_legal_moves())  # by Assumption 1

    def min_value(self, gameState, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0 or MinimaxPlayer.terminal_test(self, gameState):
            return self.score(gameState, self)  # by Assumption 2
        v = float("inf")
        for m in gameState.get_legal_moves():
            v = min(v, self.max_value(gameState.forecast_move(m),depth-1))
        return v

    def max_value(self,gameState,depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0 or MinimaxPlayer.terminal_test(self, gameState):
            return self.score(gameState, self)  # by assumption 2
        v = float("-inf")
        for m in gameState.get_legal_moves():
            v = max(v, self.min_value(gameState.forecast_move(m),depth-1))
        return v


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # TODO: finish this function!

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = AlphaBetaPlayer.get_best_move(game)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            depth = 1
            while depth >= 0:
                best_move = self.alphabeta(game, depth)
                depth += 1
        except SearchTimeout:
            return best_move  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        v = float('-inf')
        best_move = AlphaBetaPlayer.get_best_move(game)
        if bool(game.get_legal_moves()):
            for m in game.get_legal_moves():
                new_value = self.min_value(game.forecast_move(m), depth-1, alpha, beta)
                if new_value > v:
                    v = new_value
                    best_move = m
                alpha = max(alpha, v)
        return best_move

    @staticmethod
    def terminal_test(self, gameState):
        """ Return True if the game is over for the active player
        and False otherwise.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        return not bool(gameState.get_legal_moves())  # by Assumption 1

    def min_value(self, gameState, depth, alpha, beta):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0 or AlphaBetaPlayer.terminal_test(self, gameState):
            return self.score(gameState, self)  # by Assumption 2
        v = float("inf")
        for m in gameState.get_legal_moves():
            v = min(v, self.max_value(gameState.forecast_move(m), depth - 1, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def max_value(self, gameState, depth, alpha, beta):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0 or AlphaBetaPlayer.terminal_test(self, gameState):
            return self.score(gameState, self)  # by assumption 2
        v = float("-inf")
        for m in gameState.get_legal_moves():
            v = max(v, self.min_value(gameState.forecast_move(m), depth - 1, alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
