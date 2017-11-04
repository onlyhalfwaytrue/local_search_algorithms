import numpy as np
import random
import itertools;
from copy import deepcopy
from math import exp
from operator import itemgetter;
import time

class Node:
    def __init__(self, value, y, x, visited=False, minDistance=-1, pathTo = [0,0], maxLegal = -1):
        self.value = value
        self.y = y
        self.x = x
        self.visited = visited
        self.minDistance = minDistance
        self.maxLegal = maxLegal

    def __repr__(self):
        return "%s" % id(self)
    def printPuz(self):
        pass
global puzSize
def main():
    puzSize = getSizeFromFile('puzzleSize.txt') # we assume you arent erasing our puzzleSize.txt
    goodInput = False
    puzzle = makePuzzle(puzSize)
    puzzleExists = False
    while(True):
        while(goodInput == False):
            goodInput = True
            x = input('Enter number of your desired option \n\n1. Generate random puzzle with puzzleSize.txt and evaluate it\n2. Evaluate a custom puzzle from the test.txt file\n3. Perform hill climbing\n4. Perform hill climbing with RANDOM RESTARTS, starts with random generated puzzle\n5. Perform hill climbing with RANDOM WALK\n6. Perform simulated annealing\n7. Perform genetic algorithm\n8. End.\n')
            if(x == '1'): #this is already made above
                puzzleExists = True
                w = BFSevaluate(puzzle)
                print('Puzzle:\n')
                printPuzzle(puzzle)
                print('\nMinimum Moves to Each Cell:\n')
                printPuzzle(puzzle, True)
                print('\nThe score for this puzzle is ' + str(w))
            if(x == '2'):
                puzzle = getPuzzleFromFile('test.txt')
                puzzleExists = True
                w = BFSevaluate(puzzle)
                print('Values:\n')
                printPuzzle(puzzle)
                print('\nMinimum Moves to Each Cell:\n')
                printPuzzle(puzzle, True)
                print('\nThe score for this puzzle is ' + str(w))
            if(x == '3'):
                xx = input('\nEnter number of hill climb iterations\n')
                if(puzzleExists == True):
                    hillClimbEval(int(xx),puzzle)
                else:
                    hillClimbEval(int(xx))
            if(x == '4'):
                xx = input('\nEnter number of restarts\n')
                b = input('Enter steps per restart')
                restartClimb(int(xx), int(b))
            if(x == '5'):
                s = input('Enter number steps per walk:')
                p = input('\nEnter probability factor as decimal between 0 and 1. ex: 0.554\n')

                hillClimbRW(int(s), float(p), puzzle)
            if(x == '6'):
                ii = input('\nEnter number of iterations:')
                t = input('\nEnter initial temperature')
                d = input('\nEnter temperature decay rate')
                simulatedAnnealing(int(ii), int(t), float(d), puzzle) #should temp be a float?
            if(x == '7'):
                steps = input('\nEnter number of steps:')
                pop = input('\nEnter parent population')
                iPop = input('\nEnter initial population')
                selPressure = input('\nEnter selection pressure coefficient (0<x<=1)')
                mutProb = input('\nEnter mutation probability (0<=x<=1)')
                crossProb = input('\nEnter crossover probability (0<=x<=1)')
                #initial population, selection pressure, mutation prob, cross prob
                geneticAlgorithm(int(steps), int(pop), int(iPop), float(selPressure), float(mutProb), float(crossProb), puzSize) #should temp be a float?
            if (x=='8'):
                return 0
            else: goodInput = False
        goodInput = False

    #if statements for this
    ##puzzle = makePuzzle(puzSize)
    printPuzzle(puzzle) #make second arg true to print minDistances
    k = BFSevaluate(puzzle)
    print()
    printPuzzle(puzzle,True)
    print("k = " + str(k))
    k = BFSevaluate(puzzle)
    print()
    printPuzzle(puzzle,True)
    print("k = " + str(k))
    resList = hillClimbEval(500, puzzle)
    print(resList[-1])

def getPuzzleFromFile(fileName):

    with open(fileName) as f:
        M = []
        for line in f:
            line = line.split() # to deal with blank
            if line:            # lines (ie skip them)
                line = [int(i) for i in line]
                M.append(line)
    #print(M)
    dim = M[0][0]
    puzzle = np.array([[Node(0,j,i) for j in range(dim)] for i in range(dim)])
    M.pop(0)
    #print()

    #print(M)
    m = dim -1

    for i in range(dim):
        for j in range(dim):
            maxX = max(abs(m-i),abs(i))
            maxY = max(abs(m-j),abs(j))
            legal = max(maxX,maxY)
            puzzle[i][j].maxLegal = legal
            puzzle[i][j].value =  M[i][j]

    return puzzle



def getSizeFromFile(f):
    with open(f, 'r') as file:
        line = file.readline()
    n = (int (line))
    return n
def makePuzzle(n):
    puzzle = np.array([[Node(0,j,i) for j in range(n)] for i in range(n)])
    m = n - 1
    for i in range(n):
        for j in range(n):
            maxX = max(abs(m-i),abs(i))
            maxY = max(abs(m-j),abs(j))
            legal = max(maxX,maxY)
            puzzle[i][j].maxLegal = legal
            puzzle[i][j].value =  random.randint(1,legal)
            #print(str(puzzle[i][j].x) + "," + str(puzzle[i][j].y) + " : " + str(puzzle[i][j].value))

    puzzle[n-1,n-1].value = 0
    return puzzle
    #print(np.matrix(puzzle))
def printPuzzle(puzzle, dist=False):
    n = puzzle.shape[0]
    if(dist==False):
        for i in range(n):
            for j in range(n):
                print(str(puzzle[i][j].value), end='')
                if(j == n-1):
                    print('')
                else:
                    if(puzzle[i][j].value < 10):
                        print('  ', end='')
                    else: print(' ', end='')
    else:
        for i in range(n):
            for j in range(n):
                if(puzzle[i][j].minDistance == -1):
                    print("X", end='')

                else:
                    print(str(puzzle[i][j].minDistance), end='')
                if(j == n-1):
                    print('')
                else:
                    if(puzzle[i][j].minDistance < 10):
                        print('  ', end='')
                    else: print(' ', end='')

    print()

def BFStep(board, cell=[0,0]):
    out = [];
    dim = board.shape[0];
    val = board[cell[0],cell[1]].value;
    rlud = [[cell[1]+val, cell[1]-val],[cell[0]+val, cell[0]-val]];
    for thing in rlud[1]:
        if 0 <= thing and thing <dim:
            #print("thing: " + str(thing) + ", " + str(cell[1]))
            #print("vistited= " + str(board[thing, cell[1]].visited))
            if board[thing, cell[1]].visited == True:
                pass
            else:
                board[thing, cell[1]].visited = True;
                board[thing, cell[1]].minDistance = board[cell[0],cell[1]].minDistance+1;
                out.append([thing, cell[1]]);
        else:
            pass
    for item in rlud[0]:
        if 0 <= item and item <dim:

            if board[cell[0],item].visited == True:
                pass
            else:
                board[cell[0],item].visited = True;
                board[cell[0],item].minDistance=board[cell[0],cell[1]].minDistance+1
                out.append([cell[0],item]);
        else:
            pass

    return out


def BFSevaluate(board, start = [0,0]):
    dim = board.shape[0]
    currentNodes = [start];
    board[start[0],start[1]].minDistance = 0;
    board[start[0],start[1]].visited = True;
    stepsTaken = 0;
    lim = board.shape[0]^2
    while stepsTaken<2000:
        temp = [];
        for item in currentNodes:
            temp = temp + BFStep(board, item);
            #print(temp)
        #print(temp)
        currentNodes = temp;
        if currentNodes == []:
            break
        else:
            pass
        stepsTaken += 1
    notVisits = 0
    if(board[dim-1,dim-1].visited == True):
        return board[dim-1,dim-1].minDistance
    else:
        for i in range(dim):
            for j in range(dim):
                if(board[j,i].visited == False):
                    notVisits+= 1
        return -1*notVisits
def hillClimb(board):
    dim= board.shape[0]
    randX = board.shape[0]-1
    randY = board.shape[0]-1
    while(randX == dim-1 and randY == dim-1):
        randX = random.randint(0,dim-1)
        randY = random.randint(0,dim-1)
    #print(str(randX) + ", " + str(randY))
    #print(board[randX,randY].maxLegal)
    oldVal = board[randX,randY].value
    board[randX,randY].value = random.randint(1,board[randX,randY].maxLegal)
    while(oldVal == board[randX,randY].value):
        board[randX,randY].value = random.randint(1,board[randX,randY].maxLegal)
    for i in range(dim):
        for j in range(dim):
            board[i][j].visited = False
            board[i][j].minDistance = -1
    #printPuzzle(board)
def hillClimbEval(n,board = None):
    #k = BFS
    #n is iterations

    if(board is None):
        goodInput = False
        while(goodInput == False):
            goodInput = True
            x = input('\nEnter a size for the puzzle, or 0 to use size in puzzleSize.txt\n')
            if(x.isdigit()):
                a = int(x)
                if(a in [5, 7, 9, 11]):
                    board = makePuzzle(a)
                elif(a == 0):
                    puzSize = getSizeFromFile('puzzleSize.txt')
                    board = makePuzzle(puzSize)
                else:
                    goodInput = False
            else: goodInput = False
    else:
        pass
    startTime = int(round(time.time() * 1000))
    k = BFSevaluate(board)
    startK = deepcopy(k)
    oldestBoard = deepcopy(board)
    oldK = k
    l = [k]
    i = 0
    AS = 0
    while(i<n):
        oldBoard = deepcopy(board)
        hillClimb(board)
        k = BFSevaluate(board)
        if(k>=oldK):
            AS+=1
            oldK = k
            l.append(k)
            #print(list)
        else: board = oldBoard
        i+=1
    print()
    print("Iterations: " + str(i) + " Accepted steps: " + str(AS))
    print("Original board:\n")
    printPuzzle(oldestBoard)
    print("Original board's min distances from start:\n")
    printPuzzle(oldestBoard,True)
    print("Newest board:\n")
    printPuzzle(board)
    print("Newest board's min distances from start:\n")
    printPuzzle(board,True)
    print()
    #print("length: " + str(len(l)) + " " + str(m) + "\n")
    print("Every iteration's minimum distance to goal:")
    print(l)
    print("Max-min score variance: " + str(l[-1]) + " - " + str(l[0]) + " = " + str(l[-1]-l[0])+ "\n")
    endTime = int(round(time.time() * 1000))
    milisElapsed = endTime - startTime
    return (AS,startK,l[-1],milisElapsed,i)

def restartClimb(starts, steps):
    goodInput = False
    puzSize = 0
    while(goodInput == False):
        goodInput = True
        x = input('\nEnter a size for the puzzle, or 0 to use size in puzzleSize.txt\n')
        if(x.isdigit()):
            a = int(x)
            if(a in (5, 7, 9, 11)):
                puzSize = a
            elif(a == 0):
                puzSize = getSizeFromFile('puzzleSize.txt')
            else:
                goodInput = False
        else: goodInput = False

    board = makePuzzle(puzSize);
    bestBoard = deepcopy(board);
    k = BFSevaluate(board);
    startK = deepcopy(k)
    oldestBoard = deepcopy(board);
    highK=k;
    l = [];
    while starts > 0:
        j = hillClimbEval(steps, board);
        k = j[0][-1];
        l.append(j[0]);
        if k >= highK:
            bestBoard = deepcopy(j[1]);
            board = makePuzzle(puzSize);
            highK = k;
        starts -= 1;
    print()
    print("Iterations" + "Accepted steps")
    print("Original board:\n")
    printPuzzle(oldestBoard)
    print("Original board's min distances from start:\n")
    printPuzzle(oldestBoard,True)
    print("Best board:\n")
    printPuzzle(bestBoard)
    print("Best board's min distances from start:\n")
    printPuzzle(bestBoard,True)
    print()
    print('High score:' +str(k) + ', best board:')
    return (l, k, bestBoard)

def hillClimbRW(steps, prob, board = None):
    #add ssame stuff as start of the hillClimbRandom
    k = BFSevaluate(board)
    oldestBoard = deepcopy(board)
    oldK = k
    l = [k]
    i = 0
    m = 0
    while(m<steps-1):
        oldBoard = deepcopy(board)
        hillClimb(board)
        k = BFSevaluate(board)
        if(k>=oldK):
            i+=1;
            l.append(k);
            oldK = k;
            #print(list)
        else:
            r = random.uniform(0,1);
            if r <= prob:
                i+=1;
                l.append(k);
                oldK = k;
            else:
                board = oldBoard
        m+=1
    print()
    print("Original board:\n")
    printPuzzle(oldestBoard)
    print("Original board's min distances from start:\n")
    printPuzzle(oldestBoard,True)
    print("Newest board:\n")
    printPuzzle(board)
    print("Newest board's min distances from start:\n")
    printPuzzle(board,True)
    print()
    print("Every iteration's mininum distance to goal:")
    print(l)
    print("\nlength: " + str(len(l)) + ". Total tries: " + str(m) + ".")
    print()
    return (l,board)

def simulatedAnnealing(steps, initTemp, decay, board):
    k = BFSevaluate(board);
    oldestBoard = deepcopy(board);
    oldK = k;
    l =[k];
    T = initTemp;
    i=0;
    m=0;
    output = '50'
    while i <= steps:
        #if i > 13:
            #print("k = " + str(k) + " oldK = " + str(oldK) + " temp = " + str(T) + "  last 10: " + str(l[-10:]) + " m: " + str(m) + " i: " + str(i))
        oldBoard = deepcopy(board)
        hillClimb(board)
        k = BFSevaluate(board)
        if(k>=oldK):
            i+=1 #we only iterate when we get the same or better K
            l.append(k)
            oldK = k
            #print(list)
        else:
            r = random.uniform(0,1);
# 2000 iters, 1000 temp, .995 rate, try .999 next since after 55k iterations we are at 1e-120
            if(T>0):
                if((k-oldK)/T < 709.7827): #overflow error from exp(710+)
                    if r <= exp((k-oldK)/T):
                        i+=1
                        l.append(k)
                        oldK = k
                    else:
                        board = oldBoard
                else:
                    if r <= exp(709.7827):
                        i+=1
                        l.append(k)
                        oldK = k
                    else:
                        board = oldBoard
            else:
                board = oldBoard
        T*=decay;
        m+=1;
    print()
    print("Original board:\n")
    printPuzzle(oldestBoard)
    print("Original board's min distances from start:\n")
    printPuzzle(oldestBoard,True)
    print("Newest board:\n")
    printPuzzle(board)
    print("Newest board's min distances from start:\n")
    printPuzzle(board,True)
    print()
    print("length: " + str(len(l)) + ". Total tries: " + str(m) + ".")
    print(l)
    print("Max-min score variance: " + str(l[-1]) + " - " + str(l[0]) + " = " + str(l[-1]-l[0]))
    print()
    print("Start temp: " + str(initTemp) + " Decay: " + str(decay))
    return (l,board)

def boardRefresh(board):
    dim = board.shape[0];
    for i in range(dim):
        for j in range(dim):
            board[i][j].visited = False
            board[i][j].minDistance = -1

def crossover(board, board2):
    dim = board.shape[0];
    outA = np.array([[Node(0,y,x) for y in range(dim)] for x in range(dim)])
    outB = np.array([[Node(0,y,x) for y in range(dim)] for x in range(dim)])
    split = int(dim/2) + 1;
    for i in range(dim):
        for j in range(split):
            maxX = max(abs(dim-i),abs(i))
            maxY = max(abs(dim-j),abs(j))
            legal = max(maxX,maxY)
            outA[i][j].maxLegal = legal
            outB[i][j].maxLegal = legal
            outA[i][j].value = board.value
            outB[i][j].value = board2.value
        for j in range(split, dim):
            maxX = max(abs(dim-i),abs(i))
            maxY = max(abs(dim-j),abs(j))
            legal = max(maxX,maxY)
            outA[i][j].maxLegal = legal
            outB[i][j].maxLegal = legal
            outA[i][j].value = board2.value
            outB[i][j].value = board.value
    return [outA, outB]

def selection(gen2, num, selPressure):
    gen = deepcopy(gen2);
    x = 1-selPressure;
    newgen=[];
    gen.sort(key=itemgetter(0));
    gen.reverse();
    sums = [];
    if num <= len(gen):
        for i in range (len(gen)+1):
            sums.append((1-pow(x,i)))
        while len(newgen) < num:
            l=len(gen)
            r = random.uniform(0,(1-pow(x,l)));
            selected = 0;
            for n in range(l):
                if sums[n] <= r and r < sums[n+1]:
                    selected = n;
                else:
                    pass
            v = gen.pop(selected)
            newgen.append(v)
        return newgen
    else:
        return gen

def mutate(board2, mutProb):
    board = deepcopy(board2)
    dim = board.shape[0];
    cellProb = mutProb/(pow(dim, 2));
    for i in range(dim):
        for j in range(dim):
            r = random.uniform(0,1)
            if r <= cellProb:
                oldVal = board[i,j].value
                board[i,j].value = random.randint(1,board[i,j].maxLegal)
                while(oldVal == board[i,j].value):
                    board[i,j].value = random.randint(1,board[i,j].maxLegal)
    board[dim-1,dim-1].value = 0
    boardRefresh(board)
    boardRefresh(board2)



def geneticAlgorithm(steps, pop = 2, iPop = 4, selPressure = 0.5, mutProb = 1, crossProb = 0.5, psize = 5):
    generation = [];
    for i in range(iPop):
        generation.append(makePuzzle(psize))
    def evalGen(population):
        out = [];
        for thing in population:
            out.append((BFSevaluate(thing),thing))
    prevBestGuy=(0,0)
    while steps > 0:
        #selection stage with given selection pressure
        generated = evalGen(generation)
        generated.sort(key= itemgetter(0))
        bGuy = generated[-1]
        bestGuy = bGuy if bGuy[0] >= prevBestGuy[0] else prevBestGuy
        parents = selection(generated, pop, selPressure)
        offspring = [];
        #crossover stage with given crossover probability
        if pop % 2 == 0:
            for j in range(pop):
                r = random.uniform(0,1)
                if j%2 != 0:
                    pass
                elif r > crossProb:
                    offspring += [parents[j][1],parents[j+1][1]]
                else:
                    offspring += crossover(parents[j][1],parents[j+1][1])
        else:
            for j in range(1,pop):
                r = random.uniform(0,1)
                if j%2 == 0:
                    pass
                elif r > crossProb:
                    offspring += [parents[j][1],parents[j+1][1]]
                else:
                    offspring += crossover(parents[j][1],parents[j+1][1])
        #mutation stage
        generation = []
        for thing in parents:
            generation.append(thing[1])
        generation += offspring
        for thing in generation:
            mutate(thing, mutProb)
        prevBestGuy = bestGuy
        steps-=1
    lastgen = evalGen(generation)
    lastgen.sort(key=itemgetter(0));
    if bestGuy <= lastgen[-1]:
        return lastgen[-1]
    else:
        return bestGuy






if __name__ == "__main__":
    main()
