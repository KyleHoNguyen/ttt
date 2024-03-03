import numpy as np
import joblib
import os
# ------------- FUNCTIONS ----------------
def initBoard():
    # Initializes the board with 0's
    board = [0] * 9 
    printBoard(board, 0)
    return board
def placeMark(board, val, ind):
    # Places the X or O depending on player
    board[ind] = val
    printBoard(board, val)
    
    ## checks if someone won
    if(GameEnd(board)):
        return True
    return False
def printBoard(board, val):
    ## Prints board
    if(val == -1):
        print("Board State:")
    else:
        print("Your Turn:")
    counter = 0
    line = ""

    for a in board:
        if(a > 0):
            line += "[X]"
        elif(a < 0):
            line += "[O]"
        else: 
            line += "[ ]"
        counter += 1

        if(counter == 3):
            counter = 0
            print(line)
            line = ""
    print("---------")
def checkWinner(board):
    ## checks for a win
    if(board[0] == board[1] == board[2] != 0):
        return board[0]
    elif(board[3] == board[4] == board[5] != 0):
        return board[3]
    elif(board[6] == board[7] == board[8] != 0):
        return board[6]
    elif(board[0] == board[3] == board[6] != 0):
        return board[0]
    elif(board[1] == board[4] == board[7] != 0):
        return board[1]
    elif(board[2] == board[5] == board[8] != 0):
        return board[2]
    elif(board[0] == board[4] == board[8] != 0):
        return board[0]
    elif(board[2] == board[4] == board[6] != 0):
        return board[2]
    elif 0 not in board:
        # The board is full and there's no winner, indicating a tie
        return 2
    else:
        # There is no winner yet
        return 0
def GameEnd(board):
    # Prints out End statement if game is over
    val = checkWinner(board)
    if(val != 0):
        if(val == -1):
            # computer wins
            print("The Computer Wins!")
        elif(val == 1):
            # player wins
            print("The Player Wins!")
        else:
            # tie
            print("It's a Tie")
        
        return True
    else:
        return False
def select_model(model_index):
    # loads selected model
    # Define a dictionary mapping model indices to model filenames
    models_directory = 'models'
    model_filenames = {
        0: 'knn_classifier_sbc.pkl',
        1: 'knn_classifier_mbc.pkl',
        2: 'knn_classifier_fbc.pkl',
        3: 'mlp_classifier_sbc.pkl',
        4: 'mlp_classifier_mbc.pkl',
        5: 'mlp_classifier_fbc.pkl',
        6: 'svm_classifier_sbc.pkl',
        7: 'svm_classifier_mbc.pkl',
        8: 'svm_classifier_fbc.pkl',
        9: 'knn_regressor.pkl',
        10: 'linear_regressor.pkl',
        11: 'mlp_regressor.pkl',
    }
    # Check if the provided index is valid
    if model_index not in model_filenames:
        print("Invalid model index.")
        return None

    # Load the model corresponding to the provided index
    model_filename = model_filenames[model_index]
    model = joblib.load(os.path.join(models_directory, model_filename))

    return model
def print_model_options():
    # Print list of models
    model_filenames = {
        0: 'knn_classifier_sbc',
        1: 'knn_classifier_mbc',
        2: 'knn_classifier_fbc',
        3: 'mlp_classifier_sbc',
        4: 'mlp_classifier_mbc',
        5: 'mlp_classifier_fbc',
        6: 'svm_classifier_sbc',
        7: 'svm_classifier_mbc',
        8: 'svm_classifier_fbc',
        9: 'knn_regressor',
        10: 'linear_regressor',
        11: 'mlp_regressor',
    }

    print("Available model options:")
    for index, filename in model_filenames.items():
        print(f"{index}: {filename}")
    print("The best model is 3: 'mlp_classifier_sbc'")


# --------------- MAIN ---------------------

# Load model
print_model_options()
inputPos = int(input("Choose a Model Number: "))
model = select_model(inputPos)
# set up game
game = initBoard()
while not GameEnd(game):
    # gets input
    inputPos = int(input("Enter position: "))
    gameEnded = False
    if(game[inputPos] != 0):
        print("Invalid Position")
        continue
    else:
        # places X
        gameEnded = placeMark(game, 1, inputPos)
    
    # exits early if game over
    if(gameEnded):
        break
    
    # converts current game array to a 2d array
    game_numpy = np.array(game)
    game_reshaped = game_numpy.reshape(1, -1)
    # uses model to predict best position for computer
    prediction = model.predict(game_reshaped)
    #places O
    gameEnded = placeMark(game, -1, int(prediction[0]))
    
    # exits early if game over
    if(gameEnded):
        break

    

