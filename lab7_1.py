import random
from collections import defaultdict

def check_winner(board):
    wins = [
        (0,1,2), (3,4,5), (6,7,8),
        (0,3,6), (1,4,7), (2,5,8),
        (0,4,8), (2,4,6)
    ]
    for a,b,c in wins:
        if board[a] == board[b] == board[c] and board[a] != " ":
            return board[a]
    if " " not in board:
        return "draw"
    return None

def printable(board):
    return (f"{board[0]}|{board[1]}|{board[2]}\n"
            f"-+-+-\n"
            f"{board[3]}|{board[4]}|{board[5]}\n"
            f"-+-+-\n"
            f"{board[6]}|{board[7]}|{board[8]}")

class MENACE:
    def __init__(self):
        self.boxes = {}     
        self.trace = []     

    def get_box(self, board):
        """Return or initialize the matchbox for this board."""
        state = tuple(board)

        if state not in self.boxes:
            
            moves = {i: 3 for i, v in enumerate(board) if v == " "}
          
            for m in moves:
                moves[m] = max(1, moves[m])
            self.boxes[state] = moves

        return self.boxes[state]

    def choose_move(self, board):
        box = self.get_box(board)
        legal_moves = {m: b for m, b in box.items() if board[m] == " "}

        if not legal_moves:
            return None

        total = sum(legal_moves.values())
        r = random.randint(1, total)
        cumulative = 0

        for move, beads in legal_moves.items():
            cumulative += beads
            if r <= cumulative:
                self.trace.append((tuple(board), move))
                return move

    def update(self, outcome):
        rewards = {"win": 3, "draw": 1, "loss": -1}
        reward = rewards[outcome]

        for state, move in self.trace:
            box = self.boxes[state]
            box[move] = max(1, box[move] + reward)

        self.trace.clear()



def random_opponent(board):
    choices = [i for i, v in enumerate(board) if v == " "]
    return random.choice(choices) if choices else None


def play_game(menace, opponent=random_opponent, show=False):
    board = [" "] * 9
    turn = "X" 

    while True:
        if turn == "X":
            move = menace.choose_move(board)
        else:
            move = opponent(board)

        if move is None:
           
            menace.update("draw")
            return "draw"

        board[move] = turn
        result = check_winner(board)

        if result:
            if show:
                print(printable(board))
                print("Result:", result)

            if result == "X":
                menace.update("win")
                return "win"
            if result == "O":
                menace.update("loss")
                return "loss"
            menace.update("draw")
            return "draw"

        turn = "O" if turn == "X" else "X"


if __name__ == "__main__":
    menace = MENACE()

    stats = {"win":0, "loss":0, "draw":0}
    for _ in range(5000):
        stats[play_game(menace)] += 1

    print("After training:")
    print(stats)
    print("Total states learned:", len(menace.boxes))

    print("\nPlay against MENACE!")
    board = [" "] * 9
    turn = "X"

    while True:
        if turn == "X":
            move = menace.choose_move(board)
            print("\nMENACE moved to:", move)
        else:
            print(printable(board))
            human = int(input("Enter your move (0–8): "))
            while board[human] != " ":
                human = int(input("Square taken. Try again: "))
            move = human

        board[move] = turn

        result = check_winner(board)
        if result:
            print("\nFinal board:")
            print(printable(board))
            print("Result:", result)
            break

        turn = "O" if turn == "X" else "X"
