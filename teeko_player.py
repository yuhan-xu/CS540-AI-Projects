"""
Name: Yuhan Xu
Email: yxu329@wisc.edu
Class: CS 540
Project name: teeko_player.py
"""


import random
import copy


""" This class implements the game TeekoPlayer
"""
class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    """
    takes in a board state and returns a list of the legal successors
    @:param self, state
    @:return return a list of successors
    """
    def succ(self, state):
        position = []
        piece = self.my_piece  # set piece to be my_piece
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j] == piece:
                    position.append((i, j))  # append (i,j) pair to position if state[i][j] is the same as piece
        successors = []  # list for possible successors

        if len(position) == 4:  # if all pieces are dropped
            for element in position:
                # define i and j to be the first and second entry of element
                i = element[0]
                j = element[1]
                # possible moves
                moves = [[i, j + 1], [i, j - 1], [i + 1, j], [i - 1, j], [i + 1, j + 1], [i + 1, j - 1], [i - 1, j + 1],
                         [i - 1, j - 1]]
                # loop through moves
                for step in moves:
                    # if the following conditions hold
                    if 0 <= step[0] and step[0] < 5 and 0 <= step[1] and step[1] < 5 and state[step[0]][step[1]] == ' ':
                        # create a deep copy of state called new_state
                        new_state = copy.deepcopy(state)
                        new_state[step[0]][step[1]] = piece
                        new_state[i][j] = ' '  # set new_state[i][j] to be empty
                        successors.append((element, step, new_state))  # append to successors list
        else:  # otherwise
            if len(position) > 0:
                # traverse position
                for element in position:
                    i = element[0]
                    j = element[1]
                    moves = [[i, j + 1], [i, j - 1], [i + 1, j], [i - 1, j], [i + 1, j + 1], [i + 1, j - 1],
                             [i - 1, j + 1], [i - 1, j - 1]]
                    # loop through moves
                    for step in moves:
                        if 0 <= step[0] and step[0] < 5 and 0 <= step[1] and step[1] < 5 and state[step[0]][step[1]] \
                                == ' ':
                            new_state = copy.deepcopy(state)
                            new_state[step[0]][step[1]] = piece
                            successors.append((element, step, new_state))
            else:  # if position length is 0
                (row, col) = (random.randint(0, 4), random.randint(0, 4))  # randomly select position to drop
                # if state at that position is not empty, re-select another place to drop
                while state[row][col] != ' ':
                    (row, col) = (random.randint(0, 4), random.randint(0, 4))
                new_state = copy.deepcopy(state)  # create deep copy of state
                new_state[row][col] = piece  # set new_state[row][col] to be piece
                successors.append((None, (row, col), new_state))  # append to the successors list
        return successors

    """
    evaluate non-terminal states and return some float value between 1 and -1
    @:param self, state
    @:return return some float value between 1 and -1
    """
    def heuristic_game_value(self, state):
        state_copy = copy.deepcopy(state)  # create a deepcopy of the state

        if self.game_value(state_copy) != 0:  # if someone wins, return the game value of that state copy
            return self.game_value(state_copy)

        position = []  # create a empty list
        # this nested for loop append (i,j) pairs to position list when color is my_piece
        for i, z in enumerate(state_copy):
            for j, color in enumerate(z):
                if color == self.my_piece:
                    position.append((i, j))
        size = len(position)
        # calculate the average of row and col
        row_avg = sum([e[0] for e in position]) / (size + 1)
        col_avg = sum([e[1] for e in position]) / (size + 1)
        # calculate the dist of my pieces
        distance1 = sum([(e[0]-row_avg)**2 for e in position]) + sum([(e[1]-col_avg)**2 for e in position])

        position1 = []  # define a new list
        # this nested for loop append (i,j) pairs to position1  when color is opp
        for i, z in enumerate(state_copy):
            for j, color in enumerate(z):
                if color == self.opp:
                    position1.append((i, j))
        size = len(position1)
        # calculate the average of row and col in this case
        row_avg1 = sum([e[0] for e in position1]) / (size + 1)
        col_avg1 = sum([e[1] for e in position1]) / (size + 1)
        # find the dist of opp pieces
        distance2 = sum([(e[0]-row_avg1)**2 for e in position]) + sum([(e[1]-col_avg1)**2 for e in position])
        # compute the difference in reciprocal of two (distances + 1) respectively
        result = float(1 / (1 + distance1)) - float(1 / (1 + distance2))
        return result

    """
    minimax algorithm (Max_Value)
    @:param self, state, depth
    @:return return s_current, next_state, alpha
    """
    def Max_Value(self, state, depth):
        successors = self.succ(state)  # call succ to get successors of a state
        # for element in successors, if the heuristic of new_state is 1, return source_state, s, and 1
        for (source_state, s, new_state) in successors:
            if self.heuristic_game_value(new_state) == 1:
                return source_state, s, self.heuristic_game_value(new_state)

        if depth > 2:
            #  initialize source_state_curr and s_current
            source_state_curr = None
            s_current = None
            # initialize alpha to be -infinity
            alpha = float('-inf')
            for (source_state, s, new_state) in successors:
                # if heuristic_game_value of new_state is greater than alpha
                if self.heuristic_game_value(new_state) > alpha:
                    # update source_state_curr and s_current
                    source_state_curr = source_state
                    s_current = s
                    # update alpha
                    alpha = self.heuristic_game_value(new_state)
            return source_state_curr, s_current, alpha

        alpha = float('-inf')
        # initialization of s_current and next_state
        s_current = None
        next_state = None
        # loop through successors list
        for (source_state, s, new_state) in successors:
            _, _, result = self.Min_Value(new_state, depth + 1)  # get result by calling Min_Value
            if result > alpha:  # if result is greater than alpha, update alpha, current state and next state
                alpha = result
                s_current = source_state
                next_state = s
        return s_current, next_state, alpha

    """
    minimax algorithm (Min_Value)
    @:param self, state, depth
    @:return return 
    """
    def Min_Value(self, state, depth):
        successors = self.succ(state)  # call succ to get successors of a state
        # for element in successors, if the heuristic of new_state is -1, return source_state, s, and -1
        for (source_state, s, new_state) in successors:
            if self.heuristic_game_value(new_state) == -1:
                return source_state, s, self.heuristic_game_value(new_state)

        if depth > 2:  # if depth is greater than 2
            # set current source and current state to None
            source_state_curr = None
            state_current = None
            beta = float('inf')  # initialize beta to be + infinity
            # loop through successors list
            for (source_state, s, new_state) in successors:
                # if heuristic_game_value of new_state is less than beta
                if self.heuristic_game_value(new_state) < beta:
                    # assign source_state to source_state_curr, s to state_current
                    source_state_curr = source_state
                    state_current = s
                    # update beta
                    beta = self.heuristic_game_value(new_state)
            return source_state_curr, state_current, beta
        beta = float('inf')
        # set current state and next to be None
        state_current = None
        next_s1 = None
        for (source_state, s, new_state) in successors:
            # call Max_Value function to get the result
            _, _, result = self.Max_Value(new_state, depth + 1)
            if result < beta:  # if result is less than beta
                beta = result  # update beta
                state_current = source_state  # update state_current
                next_s1 = s  # update next
        return state_current, next_s1, beta

    """
    generate the subtree of depth d under this state, create a heuristic scoring function to evaluate the "leaves" at 
    depth d (as you may not make it all the way to a terminal state by depth d so these may still be internal nodes) and
    propagate those scores back up to the current state, and select and return the best possible next move using the 
    minimax algorithm.
    @:param self, state
    @:return return return the best possible next move using the minimax algorithm
    """
    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        drop_phase = True  # set drop_phase to True
        position = []  # create an empty position list

        # this nested for loop tries to append (i,j) pairs to position list if color is my_piece
        for i, z in enumerate(state):
            for j, color in enumerate(z):
                if color == self.my_piece:
                    position.append((i, j))
        # if player dropped all phases
        if len(position) == 4:
            drop_phase = False  # set drop_phase to False
        move = []  # create a new list called move
        temp_s = copy.deepcopy(state)  # create a deepcopy of state
        if not drop_phase:
            # TODO: choose a piece to move and remove it from the board
            # (You may move this condition anywhere, just be sure to handle it)
            #
            # Until this part is implemented and the move list is updated
            # accordingly, the AI will not follow the rules after the drop phase!
            source_state, s_n, _ = self.Max_Value(temp_s, 0)  # call Max_Value function to get source state, next state
            move.append(s_n)  # append next state to move
            move.append(source_state)  # append source_state to move
            return move

        # TODO: implement a minimax algorithm to play better
        _, s_n_improve, _ = self.Max_Value(temp_s, 0)
        # ensure the destination (row,col) tuple is at the beginning of the move list
        move.insert(0, s_n_improve)
        return move

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                raise Exception("You don't have a piece there!")
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row) + ": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    """
    create a function to score each of the successor states. A terminal state where your AI player wins should have the
    maximal positive score (1), and a terminal state where the opponent wins should have the minimal negative score (-1)
    @:param self, state
    @:return return 1 if AI wins, -1 if opponent win, 0 if no wins
    """
    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and 2x2 box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i + 1] == row[i + 2] == row[i + 3]:
                    return 1 if row[i] == self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col] == self.my_piece else -1

        # check \ diagonal wins
        # this nested for loop tries to find \ diagonal wins
        for row in range(2):
            for col in range(2):
                # if state[row][col] is not empty and the following conditions hold
                if state[row][col] != ' ' and state[row][col] == state[row+1][col+1] == state[row+2][col+2] == \
                        state[row+3][col+3]:
                    # if state[row][col] is my_piece return 1, otherwise, return -1
                    return 1 if state[row][col] == self.my_piece else -1
        # check / diagonal wins
        # this nested for loop tries to find / diagonal wins
        for row in range(2):
            for col in range(3, 5):
                # if state[row][col] is not empty and the following conditions hold
                if state[row][col] != ' ' and state[row][col] == state[row+1][col-1] == state[row+2][col-2] == \
                        state[row+3][col-3]:
                    # if state[row][col] is my_piece return 1, otherwise, return -1
                    return 1 if state[row][col] == self.my_piece else -1
        # check 2x2 box wins
        # this nested for loop tries to find 2x2 diagonal wins
        for row in range(4):
            for col in range(4):
                # if state[row][col] is not empty and the following conditions hold
                if state[row][col] != ' ' and state[row][col] == state[row][col+1] == state[row+1][col] == \
                        state[row+1][col+1]:
                    # if state[row][col] is my_piece return 1, otherwise, return -1
                    return 1 if state[row][col] == self.my_piece else -1

        return 0  # return 0 if no winner yet

############################################################################
    #
    # THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
    #
############################################################################

ai = TeekoPlayer()
piece_count = 0
turn = 0

# drop phase
while piece_count < 8:

    # get the player or AI's move
    if ai.my_piece == ai.pieces[turn]:
        ai.print_board()
        move = ai.make_move(ai.board)
        ai.place_piece(move, ai.my_piece)
        print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
    else:
        move_made = False
        ai.print_board()
        print(ai.opp+"'s turn")
        while not move_made:
            player_move = input("Move (e.g. B3): ")
            while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                player_move = input("Move (e.g. B3): ")
            try:
                ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                move_made = True
            except Exception as e:
                print(e)

    # update the game variables
    piece_count += 1
    turn += 1
    turn %= 2

# move phase - can't have a winner until all 8 pieces are on the board
while ai.game_value(ai.board) == 0:

    # get the player or AI's move
    if ai.my_piece == ai.pieces[turn]:
        ai.print_board()
        move = ai.make_move(ai.board)
        ai.place_piece(move, ai.my_piece)
        print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
        print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
    else:
        move_made = False
        ai.print_board()
        print(ai.opp+"'s turn")
        while not move_made:
            move_from = input("Move from (e.g. B3): ")
            while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                move_from = input("Move from (e.g. B3): ")
            move_to = input("Move to (e.g. B3): ")
            while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                move_to = input("Move to (e.g. B3): ")
            try:
                ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                 (int(move_from[1]), ord(move_from[0])-ord("A"))])
                move_made = True
            except Exception as e:
                print(e)

    # update the game variables
    turn += 1
    turn %= 2

ai.print_board()
if ai.game_value(ai.board) == 1:
    print("AI wins! Game over.")
else:
    print("You win! Game over.")

