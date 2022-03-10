from collections import Counter
from functools import reduce
from operator import itemgetter
import numpy as np
from pathlib import Path
from random import sample, random, shuffle
import pickle
import json
import time
import pprint
import ast

# The 12,972 words
words = open("words.txt", 'r').read().splitlines()
wordset = set(words)

# I ordered the words differently, interspersing 'good' and 'random' words near the beginning
zwords = open("zwords.txt", 'r').read().splitlines()

# The official words
official_goals = open("wordle_answers_alphabetical.txt", 'r').read().splitlines()
official_goal_set = set(official_goals)

best_1k = open("best_1k.txt", 'r').read().splitlines()

# Length 5 words from some Knuth project \^o^/
#sgb_words = open("sgb-words.txt", 'r').read().splitlines()
#sgb_uwords = list(filter(lambda w : w not in words, sgb_words))
#sgb_iwords = list(filter(lambda w : w in words, sgb_words))

num_words = len(words)

# I was thinking to try encoding the words via assigning a unique number to each letter and multiplying them, which wouldn't preserve ordering.
# Anyway, they are big and it was inefficient.
#primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
#alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
#alphabet_to_prime = dict(zip(alphabet, primes))

#test = "zygote"

def prime_factorize_word(word):
    prime_factors = 1
    for letter in word:
        prime_factors = prime_factors * alphabet_to_prime[letter]
    return prime_factors

#p_words = list(map(prime_factorize_word, words))
#u_words = list(set(p_words))

###
# First, I wanted to try to see whether there's a quintuple of words that solves Wordle
# So I decided to just find the best pairs (with 10 letters total), then the best triplets, and so on.
# I wasn't fully sure how I wanted to proceed, so I kinda just did things step-by-step.
# I know that theoretically, it could all be done in one process :- p
###

#s_words = list(map(set, words))
#s_words2 = list(set(map(lambda w : str(list(set(w))), words)))
sd_words = dict(map(lambda w : (w, set(w)), words))
s_words = list(sd_words.values())

best = 0
best_pairs = []
i = 0
# I don't want to actually upload this data to the github, so, sorry, it's being canceled.
if False:
    if not Path("best_pairs.txt").exists():
        with open("best_pairs.txt", "a") as f:
            for w1, s1 in sd_words.items():
                for w2, s2 in sd_words.items():
                    pair = s1.union(s2)
                    lpair = len(pair)
                    if lpair == best:
                        f.write(f"({w1}, {w2})\n")
                    elif lpair > best:
                        print("New best: %s + %s for len %s" % (w1, w2, lpair))
                        best = lpair
                        f.write(f"({w1}, {w2})\n")
                    i = i + 1
                    if i % 10000 == 0:
                        print("step %s" % i)

    if not Path("tmp.pickle").exists():
        best_pairs_dict = dict()
        with open("best_pairs.txt", "r") as f:
            for l in f.readlines():
                l2 = l.rstrip("\n").split(",")
                w1 = l2[0][1:]
                w2 = l2[1][1:-1]
                key = "".join(sorted(set(w1).union(set(w2))))
                if len(key) == 10: 
                    best_pairs.append((w1,w2))
                    best_pairs_dict[key] = (w1,w2)

    unique_best_pairs = best_pairs_dict.values()

    unique_words = dict()
    unique_best_words = dict()

    for w in words:
        key = "".join(sorted(set(w)))
        unique_words[key] = w
        if len(key) == 5:
            unique_best_words[key] = w

# Incrementally adding the data to a temporary pickle file for prototyping with "python -i words.py"
    #with open("tmp5.pickle", 'rb') as f:
    #    best_pairs_dict, unique_best_words, best_triples_dict, best_quadruples_dict, best_quintuples_dict, best_quintuples_unique_dict = pickle.load(f)

    best = 0
    i = 0
    if not Path("best_triples.txt").exists():
        with open("best_triples.txt", "a") as f:
            for k1, w1 in unique_best_words.items():
                for k2, (w2, w3) in best_pairs_dict.items():
                    triple = set(k1).union(set(k2))
                    ltriple = len(triple)
                    if ltriple == best:
                        f.write(f"{w1},{w2},{w3}\n")
                    elif ltriple > best:
                        print("New best: %s + %s + %s for len %s" % (w1, w2, w3, ltriple))
                        best = ltriple
                        f.write(f"{w1},{w2},{w3}\n")
                    i = i + 1
                    if i % 1000000 == 0:
                        print("step %s" % i)

# Turns out Python crashes with the number of triples already
# The size started decreasing after this, finally.
# Kinda fun to learn that writing data to the disk wasn't such a big slow-down, from subjective user time.
# split -d -l 10000000 best_triples.txt best_triples.txt

    if not Path("best_triples_dict00.pickle").exists(): #I'll just check one, lol
        for i in range(6):
            postfix = "0{}".format(i)
            best_triples_dict = dict()
            with open("best_triples.txt{}".format(postfix), "r") as f:
                for l in f.readlines():
                    l2 = l.rstrip("\n").split(",")
                    w1, w2, w3 = l2
                    key = "".join(sorted(set(w1).union(set(w2)).union(set(w3))))
                    if len(key) == 15:
                        best_triples_dict[key] = (w1,w2,w3)
                with open("best_triples_dict{}.pickle".format(postfix), "wb") as fp:
                    pickle.dump(best_triples_dict, fp)


    if not Path("tmp.pickle").exists():
        best_triples_dict = dict()
        for i in range(6):
            with open("best_triples_dict{}.pickle".format(postfix), 'rb') as fp:
                best_triples_dict_tmp = pickle.load(fp)
                best_triples_dict = {**best_triples_dict, **best_triples_dict_tmp}

    best = 20 #Lazy -- now that I verified this is the best ;D
    i = 0
    j = 0
    if not Path("best_quadruples.txt").exists():
        with open("best_quadruples.txt", "a") as f:
            print(f"There are {len(unique_best_words) * len(best_triples_dict):,} iterations.")
            #There are 4,615,436,394 iterations.   
            for k1, w1 in unique_best_words.items():
                for k2, (w2, w3, w4) in best_triples_dict.items():
                    i = i + 1
                    quadruple = set(k1).union(set(k2))
                    lquadruple = len(quadruple)
                    if lquadruple == best:
                        f.write(f"{w1},{w2},{w3},{w4}\n")
                        j = j+1
                    elif lquadruple > best:
                        print("New best: %s + %s + %s + %s for len %s" % (w1, w2, w3, w4, lquadruple))
                        best = lquadruple
                        f.write(f"{w1},{w2},{w3},{w4}\n")
                        j = j+1
                    if i % 10000000 == 0:
                        print(f"step {i:,} and found {j:,}")


    if not Path("tmp4.pickle").exists():
        best_quadruples_dict = dict()
        with open("best_quadruples.txt", "r") as f:
            for l in f.readlines():
                l2 = l.rstrip("\n").split(",") # w1, w2, w3, w4 = l2
                key = "".join(sorted(set(letter for word in l2 for letter in word)))
                best_quadruples_dict[key] = tuple(l2)

    best = 24 # lol, I verified this works 
    i = 0
    j = 0
    if not Path("best_quintuples.txt").exists():
        with open("best_quintuples.txt", "a") as f:
            print(f"There are {len(unique_best_words) * len(best_quadruples_dict):,} iterations.")
            #There are 181,919,292 iterations.
            for k1, w1 in unique_best_words.items():
                for k4, (w2, w3, w4, w5) in best_quadruples_dict.items():
                    i = i + 1
                    quintuple = set(k1).union(set(k4))
                    lquintuple = len(quintuple)
                    if lquintuple == best:
                        f.write(f"{w1},{w2},{w3},{w4},{w5}\n")
                        j = j+1
                    elif lquintuple > best:
                        print("New best: %s + %s + %s + %s + %s for len %s" % (w1, w2, w3, w4, w5, lquintuple))
                        best = lquintuple
                        f.write(f"{w1},{w2},{w3},{w4},{w5}\n")
                        j = j+1
                    if i % 10000000 == 0:
                        print(f"step {i:,} and found {j:,}")

    if not Path("tmp5.pickle").exists():
        best_quintuples_dict = dict()
        with open("best_quintuples.txt", "r") as f:
            for l in f.readlines():
                l2 = l.rstrip("\n").split(",")
                #w1, w2, w3, w4 = l2
                key = "".join(sorted(set(letter for word in l2 for letter in word)))
                best_quintuples_dict[key] = best_quintuples_dict.get(key, []) + [tuple(l2)]

# I think these shortcuts were ultimately deprecated 🤷
def compress(string):
    return "".join(sorted(string))

def decompress(string):
    word_list = [] 
    for i in range(0,5):
        word_list.append(string[i*5:(i+1)*5])
    return word_list

def decompresst(string): # wololo
    return (string[0:5], string[5:10], string[10:15], string[15:20], string[20:25])

# I wonder what the "standard" view on using so many "if False:" statements is . . . 🤔
if False:
    if not Path("best_quints_25.txt").exists():
        with open("best_quints_25.txt", "w") as f:
            for key in best_quintuples_unique_dict.keys():
                if len(key) == 25:
                    f.write(f"{key}:\n\n")
                    for word_list in best_quintuples_unique_dict[key]:
                        f.write("    %s %s %s %s %s\n" % word_list)
                    f.write("\n\n")

# The following code is to check if a good quntiple of words solves every single target.
# I've included the few functions it uses a bit out of place just in case you want to try running it :- p

# Given a word and a goal, filter the list of possible words according to what the hint would be.
# Note that it may not treat yellow letters correctly in the case that there are 2+ in a word.  (Fixed later, at least.)
def filter_word_by_goal2(word, goal, possible_words):
    for index in range(0,5):
        letter = word[index]
        if letter == goal[index]: # Keep words where greens match
            possible_words = list(filter(lambda word : word[index] == letter, possible_words))
        elif letter in goal: # 1) Filter out words with yellow in matching index and 2) Keep words with the yellow letter elsewhere 
            possible_words = list(filter(lambda word : (word[index] != letter) and (letter in word), possible_words))
        else: # Remove all words containing grey letters.
            possible_words = list(filter(lambda word : letter not in word, possible_words))
    return possible_words

# Takes a list of words, possible words, and a goal for hint generation.
# Returns the words that have not yet been ruled out (ideally only the goal).
def filter_solution_by_goal(solution, goal, possible_words):
    for word in solution:
        possible_words = filter_word_by_goal2(word, goal, possible_words)
    return possible_words

if False:
    broken = []
    winners = []
    stats = []
    killers = Counter() 
    total = len(words)
    possible_solutions = [solution for wordset in best_quintuples_unique_dict.values() for solution in wordset]
    for solution in possible_solutions:
        solved = 0
        failed = 0
        options_left = 0
        wtf = [] ## It's empty. -- debugging variable
        success = True
        for goal in words:
            possible_words = filter_solution_by_goal(solution, goal, words)
            num_possible_words = len(possible_words)
            if num_possible_words == 1:
                if possible_words[0] == goal:
                    solved = solved + 1
                    options_left = options_left +  1
                else: # Just making sure ;D
                    wtf.append(goal)
                    print(f"{goal} is not {yellow_filtered_words[0]}!")
            else:
                killers[goal] = killers[goal] + 1
                failed = failed + 1
                options_left = options_left + num_possible_words
                if success:
                    broken.append((solution, goal, possible_words))
                    print(f"{goal} breaks {solution} with {num_possible_words} possibilities.")
                    success = False
        if success:
            winners.append(solution)
            print(f"{solution} is victorious!")
        winrate = solved / total
        options_left = options_left / total
        print(f"{solution} has winrate {winrate} with {solved} solved and {failed} failures with on average {options_left} remaining possible words.")
        stats.append((solved, failed, winrate, options_left, solution))

###
#  Next up is code for dealing with Wordle in more general terms. 
###

# Wow!  4-5 seconds faster with possible_words=words!  
def filter_word_by_goal3(word, goal, possible_words):
    filters = []
    for index in range(0,5):
        letter = word[index]
        if letter == goal[index]: # Keep words where greens match
            filters.append((0,index,letter))
        elif letter in goal: # 1) Filter out words with yellow in matching index and 2) Keep words with the yellow letter elsewhere 
            filters.append((1,index,letter))
        else: # Remove all words containing grey letters.
            filters.append((2,index,letter))
    for t, index, letter in sorted(filters):
        if t == 0:
            possible_words = list(filter(lambda word : word[index] == letter, possible_words))
        elif t == 1:
            possible_words = list(filter(lambda word : (word[index] != letter) and (letter in word), possible_words))
        else:
            possible_words = list(filter(lambda word : letter not in word, possible_words))
    return possible_words

# Turns out it's more important to work with hints directly.
# Focus on the information.
# And the exact scheme Wordle uses for hints is a bit confusing (which I figured out via trial and error xD)
# If you over-guess a letter, you'll only receive hints for the number of the letter in the word.
def get_hints3(word, goal):
    hints = ['', '', '', '', '']
    letter_counts = Counter(goal)
    for index in range(0,5):
        letter = word[index]
        if letter == goal[index]: 
            hints[index] = 'g'
            letter_counts[letter] -= 1
    for index in range(0,5):
        if not hints[index]:
            letter = word[index]
            if letter_counts[letter] > 0 and letter in goal: 
                hints[index] = 'y'
                letter_counts[letter] -= 1
            elif not hints[index]: 
                hints[index] = 'b'
    return "".join(hints)

# I tried mapping hints to ints in a way that seemed similar to 3b1b
# But it is a bit slower, so not worth it if I don't actually use the matrices intelligently.
#hintarr = np.load("hintarr.npy")
color_to_int = {'g' : 2, 'y' : 1, 'b': 0}
int_to_color = {2: 'g', 1 : 'y', 0: 'b'}

def hint_to_int(c):
    if c == 'g':
        return 2
    if c == 'y':
        return 1
    return 0

def get_hints4(word, goal):
    hint = get_hints3(word,goal)
    return sum((3**i) * color_to_int[c] for i, c in enumerate(hint))

def int_to_hint(i):
    hints = []
    for x in range(5):
        hints.append(int_to_color[i % 3])
        i = i // 3
    return "".join(hints)

def get_hints5(word, goal):
    return int_to_hint(hintarr[worddic[word]][worddic[goal]])

# Building said array of all hint pairs
if False:
    hintarr = np.zeros((num_words, num_words), dtype=np.uint8)
    for i1 in range(num_words):
        word1 = words[i1]
        #worddic[i1] = word1
        for i2 in range(num_words):
            word2 = words[i2]
            hintarr[i1][i2] = get_hints4(word1, word2)
    np.save("hintarr.npy", hintarr)

#hintarr = np.load("hintarr.npy")


#>>> start = time.time(); len([get_hints5(w, 'pause') for w in words]); end = time.time(); print(f"{end - start}")
#0.13312935829162598
#>>> start = time.time(); len([get_hints3(w, 'pause') for w in words]); end = time.time(); print(f"{end - start}")
#0.08193016052246094

# Some hacky ideas I had that didn't quite survive the test of time:
# Also functions superceded 
#def filter_word_by_hints(word, received_hint, possible_words):
#def get_different_word(word, hints, possible_words):
#def get_different_wordset(word, hints, possible_words):
#def obtain_word_possibilities(word, possible_words):
#def calculate_word_possibilities3(word, possible_words):
#def calculate_pair_possibilities3(word1, word2, possible_words):

# Deprecated.  
# Finds the worst-case hint-bucket (and computes this for EACH goal in the bucket).
# Also computes the average or expected hint-bucket size.
def locate_word_possibilities(word1, possible_words):
    accumulator = 0
    worst = 0
    for goal in possible_words:
        l = len(filter_word_by_goal3(word1, goal, possible_words))
        accumulator += l
        worst = max(worst, l) 
    return worst, accumulator / len(possible_words)

# A score function adapted from https://github.com/Ermine516/GenericWordleSolvability 
dscores = {'g' : 2, 'y' : 1, 'b' :0}

def dscore(hints):
    return 1 + (10 - (dscores[hints[0]] + dscores[hints[1]] + dscores[hints[2]] + dscores[hints[3]] + dscores[hints[4]])) / 100

#def calculate_word_possibilities3_with_dscore(word, possible_words):
#def calculate_pair_possibilities3_with_dscore(word1, word2, possible_words):

###
# The primary utility functions for working with move simulations.
###

# WAY faster than locate_word_possibilities!
# Just use a dictionary or Counter to batch hints and the word filtering happens naturally.
def get_hint_distribution(word, possible_words):
    hints = Counter()
    for goal in possible_words:
        hint = get_hints3(word, goal)
        hints[hint] += 1
    return hints

# And then I can get the best word thanks to the Counter
def score_hint_distribution(word, possible_words):
    return get_hint_distribution(word, possible_words).most_common(1)[0][1]

# Expected value of {score * hint-bucket size}
def dscore_hint_distribution(word, possible_words):
    hints = get_hint_distribution(word, possible_words)
    total = sum(hints.values())
    acc = 0
    for hint, count in hints.items():
        score = dscore(hint)
        acc += score * (count**2)
    return acc / total

# Gather goals into hint-buckets.
def get_hint_distribution_with_words(word, possible_words):
    hints = dict()
    for goal in possible_words:
        hint = get_hints3(word, goal)
        hints[hint] = hints.get(hint, []) + [goal]
    return hints

# Not used
#def get_hint_distribution_with_words_as_sets(word, possible_words):

# Do two-step hint buckets for lookahead (ultimately not used much).
def get_pair_distribution_with_words(word1, word2, possible_words):
    hint_dist1 = get_hint_distribution_with_words(word1, possible_words)
    hint_dist2 = get_hint_distribution_with_words(word2, possible_words)
    hints = dict()
    for goal in possible_words:
        hint1 = get_hints3(word1, goal)
        hint2 = get_hints3(word2, goal)
        nhint = hint1+hint2
        tmp = hints.get(nhint, set())
        tmp.update(set(hint_dist1[hint1]) & set(hint_dist2[hint2]))
        hints[nhint] = tmp
    return hints

# Now count it!
def count_pair_distribution_with_words(word1, word2, possible_words):
    hints = get_pair_distribution_with_words(word1, word2, possible_words)
    counts = Counter()
    for h, s in hints.items():
        counts[h] = len(s)
    return counts

# And score it ~ 
def score_pair_distribution(word1, word2, possible_words):
    counts = count_pair_distribution_with_words(word1, word2, possible_words)
    return counts.most_common(1)[0][1]

# The dscore version
def dscore_pair_distribution(word1, word2, possible_words):
    hint_dist1 = get_hint_distribution_with_words(word1, possible_words)
    hint_dist2 = get_hint_distribution_with_words(word2, possible_words)
    hints = dict()
    acc = 0; total = 0
    for goal in possible_words:
        hint1 = get_hints3(word1, goal)
        hint2 = get_hints3(word2, goal)
        nhint = hint1+hint2
        tmp = hints.get(nhint, set())
        wordlist = set(hint_dist1[hint1]) & set(hint_dist2[hint2]); lwdl = len(wordlist)
        tmp.update(wordlist)
        hints[nhint] = tmp
        acc += dscore(hint1) * dscore(hint2) * (lwdl**2); total += lwdl
    return acc / total

# Looking at a partial comparison of "best first two moves".  As it's non-total, it was easy for me to find a better solution, replacing  'irons' with 'irony'. 
#[(129, 'salet', 'irons'), (132, 'salet', 'noirs'), (132, 'salet', 'noris'), (132, 'solei', 'rants'), (132, 'solei', 'tarns'), (134, 'nates', 'reoil'), (134, 'tales', 'irons'), (136, 'setal', 'irons'), (136, 'setal', 'noris'), (144, 'nates', 'loirs'), (144, 'nates', 'loris'), (144, 'nates', 'roils'), (144, 'tales', 'noirs'), (144, 'tales', 'noris'), (145, 'canes', 'reoil'), (147, 'cates', 'irons'), (147, 'nates', 'soral'), (147, 'rates', 'sloan'), (147, 'rates', 'solan'), (147, 'setal', 'noirs'), (147, 'stoae', 'nirls'), (147, 'taces', 'irons'), (147, 'tares', 'sloan'), (147, 'tares', 'solan'), (149, 'cates', 'reoil'), (149, 'taces', 'reoil'), (152, 'canes', 'loirs'), (152, 'canes', 'loris'), (152, 'canes', 'roils'), (152, 'reais', 'sloan'), (152, 'solei', 'darts'), (155, 'slate', 'irons'), (155, 'slate', 'noirs')]
# Best serai: [(174, 'serai', 'loans'), (189, 'serai', 'loast'), (189, 'serai', 'lotsa'),

#>>> score_pair_distribution('salet', 'irony', words)
#127

# Counts for hint-buckets of 'salet'.  Might be amusing to reference.
#Counter({'bbbbb': 865, 'ybbbb': 824, 'bbbyb': 715, 'bybbb': 583, 'bbbgb': 552, 'bgbbb': 441, 'ybbyb': 401, 'ygbbb': 393, 'ybbgb': 364, 'yybbb': 340, 'bybyb': 301, 'gbbbb': 269, 'bbbby': 269, 'bbybb': 263, 'byybb': 256, 'ybbby': 256, 'bbyyb': 208, 'bgbgb': 198, 'bbbyy': 195, 'ybybb': 173, 'gybbb': 167, 'gbbyb': 151, 'bbygb': 147, 'gbbgb': 129, 'byyyb': 129, 'ybbyy': 126, 'bybby': 125, 'ygbby': 123, 'bbbbg': 119, 'ygbgb': 119, 'bgbby': 118, 'bbbgy': 113, 'yybyb': 111, 'bgbyb': 101, 'bbgbb': 100, 'yyybb': 99, 'ybgbb': 98, 'gbbby': 95, 'bgybb': 92, 'ybyyb': 87, 'yybby': 85, 'bybyy': 81, 'bybbg': 81, 'gbybb': 73, 'ggbbb': 73, 'ybbgy': 72, 'bygbb': 68, 'gbbyy': 64, 'ybygb': 64, 'ygybb': 63, 'bbbyg': 62, 'gybyb': 60, 'bybgb': 58, 'bbgyb': 57, 'bggbb': 57, 'bbggb': 56, 'ybbbg': 53, 'gybby': 52, 'gyybb': 48, 'bbbgg': 44, 'gbyyb': 44, 'gbbbg': 43, 'yggbb': 41, 'bgbbg': 37, 'bgbgy': 37, 'bbyyy': 37, 'ybggb': 37, 'bbyby': 36, 'yybgb': 35, 'ybgyb': 35, 'bgygb': 34, 'yyyyb': 34, 'yybyy': 33, 'ggbgb': 32, 'byyby': 31, 'bbybg': 30, 'ybbyg': 30, 'bgyyb': 30, 'bybyg': 27, 'gbgbb': 27, 'ybyby': 25, 'ygbyb': 23, 'ygbgy': 23, 'ybgby': 22, 'bgbyy': 20, 'gybbg': 19, 'byybg': 19, 'gybyy': 19, 'bgyby': 18, 'yygbb': 18, 'yybbg': 16, 'bygyb': 16, 'gbygb': 16, 'gbbgy': 15, 'bbgby': 15, 'gyyyb': 14, 'bggby': 14, 'bgggb': 14, 'gbyby': 13, 'gggbb': 13, 'bbyyg': 13, 'bbygy': 13, 'ygygb': 13, 'byygb': 12, 'ygggb': 12, 'ybgyy': 12, 'byyyy': 11, 'byggb': 11, 'gbggb': 11, 'bbgyy': 10, 'ygbyy': 10, 'ggbby': 10, 'byyyg': 9, 'ybbgg': 9, 'bbggg': 9, 'gbbyg': 9, 'yyyby': 9, 'gygbb': 8, 'ggbyb': 8, 'yggby': 8, 'ygbbg': 7, 'yybyg': 7, 'bybgy': 7, 'bgbgg': 7, 'ybyyy': 7, 'bgybg': 6, 'bggyb': 6, 'gbybg':6, 'gbyyy': 6, 'bbgyg': 6, 'ygyby': 6, 'gbgyb': 5, 'ygyyb': 5, 'ggbbg': 5, 'gyyyy': 5, 'yyybg': 5, 'ggybb': 5, 'bgygy': 5, 'gyybg': 5, 'bbggy': 5, 'bygby': 5, 'bygyy': 4, 'bbgbg': 4, 'gbyyg': 4, 'gyyby': 4, 'bbygg': 4, 'bgyyy': 4, 'gggyb': 4, 'yyygb': 4, 'ybygy': 4, 'ybggy': 4, 'gbbgg': 3, 'gybyg': 3, 'yggyb': 3, 'byygy': 3, 'gggby': 3, 'yyggb': 3, 'yybgy': 3, 'yyyyy': 3, 'ggbgy': 3, 'gygyb': 3, 'gybgb': 3, 'yybgg': 2, 'bgggg': 2, 'byggg': 2, 'bybgg': 2, 'yygby': 2, 'ybybg': 2, 'yygyb': 2, 'gggbg': 2, 'ggggb': 2, 'gbgby': 2, 'bgggy': 2, 'ybggg': 1, 'ggbyy': 1, 'bygyg': 1, 'gbygg': 1, 'gygbg': 1, 'gbygy': 1, 'yyyyg': 1, 'gbgbg': 1, 'bygbg': 1, 'ybyyg': 1, 'bgbyg': 1, 'ygbyg': 1, 'bggbg': 1, 'bgygg': 1, 'ggyyb': 1, 'ggggg': 1, 'ggygb': 1, 'ggybg': 1, 'gyygb': 1, 'ygyyy': 1, 'ygggy': 1})

count = 0

# Greedily compute solutions for every goal.  
# If no limit is set, it will just keep going until it wins, which isn't so bad.
# bword is the move to play in move=num and bwords are the moves that need to be solved
def compute_all_wordles(bword, bwords, move=1, by_dscore=False, lookahead=False, history=[], limit=69):
    global count
    if move >= limit:
        print(f"Limit {limit} exceeded with {bword}:{bwords[:5]}.")
        return (history + bwords, move + len(bwords))
    solutions = dict()
    goals = get_hint_distribution_with_words(bword, bwords)
    for hint, wordlist in goals.items():
        if len(wordlist) == 1:
            count = count + 1
            if bword == wordlist[0]:
                solutions[hint] = (history + [bword], move)
                if move > 6:
                    print(f"Solved {hint}:{wordlist[0]} on move {move} with guesses: {history}.")
            else:
                solutions[hint] = (history + [bword] + wordlist, move+1)
                if move > 5:
                    print(f"Solved {hint}:{wordlist[0]} on move {move+1} with guesses: {history + [bword]}.")
            if count % 100 == 0:
                print(f"There are {count} solved words.")
            continue
        elif move == 1 and lookahead:
            if by_dscore:
                next_word = sorted([(dscore_pair_distribution(word, word2, wordlist), word) for word in words for word2 in wordlist])[0][1]
            else:
                next_word = sorted([(score_pair_distribution(word, word2, wordlist), word) for word in words for word2 in wordlist])[0][1]
                #next_word = sorted([(score_pair_distribution(word, word2, wordlist), word) for word in (['irony'] + best_1k + wordlist) for word2 in wordlist])[0][1]
        elif by_dscore:
                next_word = sorted([(dscore_hint_distribution(word, wordlist), word) for word in words])[0][1]
        else:
            next_word = sorted([(score_hint_distribution(word, wordlist), word) for word in words])[0][1]
        solutions[hint] = compute_all_wordles(next_word, wordlist, move=move+1, history=(history+[bword]), limit=limit, by_dscore=by_dscore)
        # Oh, lol, I could check-solutions, here, and try all next words... >:D.
        # Or at least, try maybe 10 next words before giving up and going to the next level >:D
        # -- Oh, lol, I was realizing I might wanna throw brute-force search in!
        # Well, that's below
    return solutions

# Set up: bwords = official and guesses = full list, so it should always include and be bigger than the official.
# Meh, just deprecate this.
#def compute_all_wordles_hard(bword, bwords, guesses=None, move=1, extra=False, lookahead=1, history=[]):

# Calculating the word distributions for starter words.
#word_distros = dict()
#for i, word in enumerate(words):
#  c = get_hint_distribution(word, words)
#  d = get_hint_distribution_with_words_as_sets(word, words)
#  word_distros[word] = (c, d)
# 833Mb!
#wc_counts = sorted([(c.most_common(1)[0][1] , word) for word, (c, _) in word_distros.items()])
#>>> wc_counts[:10]
#[(697, 'serai'), (769, 'reais'), (769, 'soare'), (776, 'paseo'), (801, 'aeros'), (821, 'kaies'), (823, 'nares'), (823, 'nears'), (823, 'reans'), (825, 'stoae')]
#>>> wc_counts[-1]
#(8189, 'gyppy')

# Looking at the 'best' words according to possible word splitting and a dscore metric. 
# It's pretty similar but a biiiit different.
#>>> dscore_distros = {}
#>>> for word in words:
#    ...  dscore_distros[word] = dscore_hint_distribution(word, words)
#    >>> sc = sorted([(s, w) for w,s in dscore_distros.items()])
#>>> sc[:10]
#[(2375.1298951588037, 'lares'), (2403.1796176379894, 'rales'), (2501.012565525748, 'reais'), (2503.9330866481655, 'tares'), (2510.490749306198, 'nares'), (2511.9708603145236, 'aeros'), (2513.8434320074007, 'soare'), (2575.955673758865, 'rates'), (2603.6278137526983, 'arles'), (2653.5081714461917, 'aloes')]
#>>> sc[-10:]
#[(48949.254239901325, 'jugum'), (49075.940178846744, 'yukky'), (49280.66057662658, 'bubby'), (49350.35800185014, 'cocco'), (50011.36578785076, 'fuzzy'), (51477.88629355535, 'immix'), (51865.8388066605, 'hyphy'), (53230.85892691952, 'xylyl'), (53373.009019426456, 'gyppy'), (53518.9004008634, 'fuffy')]

# I tried to see if referencing these word distrubions would speed things up.
# But it didn't because I'm often working with subets of the full words list, which would require additional computations anyway
#def score_hint_distribution2(word):
#    return word_distros[word][0].most_common(1)[0][1]

#def get_hint_distribution2(word):
#    return word_distros[word][1]

###
# Utility functiosn for dealing with Wordle solution dictionaries
###

# Checks a solution dictionary to see if it goes over the limit anywhere
def check_results(solutions, limit=6):
    if isinstance(solutions, tuple):
        if solutions[1] > limit:
            #print(f"Failed check: {solutions}")
            return False
        else:
            return True
    for _, v in solutions.items():
        if not check_results(v):
            return False
    return True

def retrieve_all_results(solutions):
    if isinstance(solutions, tuple):
        return [solutions]
    results = []
    for _, v in solutions.items():
        results += retrieve_all_results(v)
    return results

# Return all leaves above the limit, 6 for normal Wordle.
def retrieve_false_results(solutions, hints = [], limit=6):
    results = []
    if isinstance(solutions, tuple):
        if solutions[1] > limit:
            return [(hints, solutions)]
        return []
    for h, v in solutions.items():
        results += retrieve_false_results(v, hints + [h])
    return results

# Sometimes I just want to access an actual leaf of the tree and that's it!  =D
def retrieve_first_result(solutions):
    if isinstance(solutions, tuple):
        return solutions
    return retrieve_first_result(next(iter(solutions.values())))

# Some functions didn't handle the starter word correctly in the history.
# Irrelevant to actually solving it but ugly.
def fix_results(solutions):
    if isinstance(solutions, tuple):
        wordlist, count = solutions
        if wordlist[-2] == wordlist[-1]:
            wordlist.pop()
            return (wordlist, count - 1)
        else: 
            return solutions
    for hint, next_solutions in solutions.items():
        solutions[hint] = fix_results(next_solutions)
    return solutions

# Sloppy code repair =D
def fixup_history(solutions, history):
    if isinstance(solutions, tuple):
            return (solutions[0], solutions[1], history + solutions[2])
    for hint, next_solutions in solutions.items():
        solutions[hint] = fixup_history(next_solutions, history)
    return solutions

# Convert between 'new' and 'old' solution formats.
# The new is better and what the brute-force method creates,
# But a lot of utility functions only work on the 'old' format.
def translate_res_new_to_old_rec(solutions):
    if isinstance(solutions, tuple):
            return (solutions[2], solutions[0])
    for hint, next_solutions in solutions.items():
        solutions[hint] = translate_res_new_to_old_rec(next_solutions)
    return solutions

def translate_res_new_to_old(solutions):
    translate_res_new_to_old_rec(solutions)

def translate_res_old_to_new_rec(solutions):
    if isinstance(solutions, tuple):
            return (solutions[1], solutions[0][-1], solutions[0])
    for hint, next_solutions in solutions.items():
        solutions[hint] = translate_res_old_to_new_rec(next_solutions)
    return solutions

def translate_res_old_to_new(solutions):
    translate_res_old_to_new_rec(solutions)

# Retrieves all words in a (sub)dictionary from a solution.  Utility function.
def get_sol_entries(d):
    return [wl[-1] for wl, _ in retrieve_all_results(d)]

# Locate the hint where a word lies in a solution: 
#find_sol_hint('nadir', swift_solution_full) --> 'bbybb'
def find_sol_hint(w, d):
    for key, value in d.items():
        if w in get_sol_entries(value):
            return key

# Given an old-solution, return a dictionary consting of tuples, (move, next_possibilities).
# Basically removing helpful but unnecessaryr information from the barebones solution.
def get_sol_skeleton(d, i=1):
    moves = dict()
    for hint, next_solutions in d.items():
        res = retrieve_first_result(next_solutions)[0]
        if hint != "ggggg" and len(res) > i:
            next_move = res[i]
            if isinstance(next_solutions, dict):
                moves[hint] = (next_move, get_sol_skeleton(next_solutions, i+1))
            else:
                moves[hint] = next_move
        else:
            moves[hint] = res[-1]
    return moves

# Invokes the recursive function above with the first move.
def wrap_skel(d):
    return (retrieve_first_result(d)[0][0], get_sol_skeleton(d))

# Plays out a history or solution of moves and returns the resulting dictionary of hints and their buckets
def play_history(history, possible_words):
    hint_dist = get_hint_distribution_with_words(history[0], possible_words)
    history = history[1:]
    if history:
        sol = dict()
        for key, value in hint_dist.items():
            if len(value) > 1:
                sol[key] = play_history(history, value)
            else:
                sol[key] = value
        return sol
    else:
        return hint_dist

# Given the output of play_history, flattens the dictionary
# Returns tupless: (size_of_bucket, words_in_bucket)
# e.g., get_history_buckets(play_history(['swift', 'lover', 'tanty', 'aahed'], words))
# [(4, ['cigar', 'cimar', 'imbar', 'quair']), (1, ['chair']), ...] 
def get_history_buckets(history_dic):
    if isinstance(history_dic, list):
        return [(len(history_dic), history_dic)]
    results = []
    for value in history_dic.values():
        results += get_history_buckets(value)
    return results

# Retrieve the words and hints on which an attempted solution (old) fails.  Hardcoded to k = 6
def get_fails(partial, official_goals_only=False):
    failed4 = set(); fails = dict()
    for h, w in partial.items():
        f = retrieve_false_results(w)
        if f:
            fset = set([w[-1] for _, (w,n) in f])
            if official_goals_only and fset & official_goal_set:
                fails[h] = set([w[-1] for _, (w,n) in f])
                failed4.update(fset)
            else:
                fails[h] = set([w[-1] for _, (w,n) in f])
                failed4.update(fset)
    return fails, failed4

# Relic from when I wanted to selectively try the greedy solution some extra times (without actually brute-forcing)
#def try_again(fails, partials, base=None, trynum=0, extra=False, official_goals_only=False, history=['salet']):

# Take one of a wordle assist function where I had to manually keep track of the remaining words and the history 
#def wordle_assist(guess, hint, remaining_words, move2="", num=3):

# Some good words
#good_words = ["lurid", "poise", "roast", "caste", "adore", "death", "gymps", "nymph", "fjord", "vibex", "waltz"
#             ,"gucks", "vozhd", "waqfs", "treck", "jumpy", "bling", "brick", "clipt", "kreng", "prick", "pling"
#             ,"bemix", "clunk", "grypt", "xylic", "cimex", "brung", "blunk", "kempt", "quick", "verbs", "arose"]

# A one-off greedy Wordle solver using a bit of hacks, such as this "good words" list.
# It got around 97.4% on the full dataset
#def solve_wordle(seed, goal, possible_words, seed2=None, sample_num=0):

# I spent some time trying to analyze *ESTS and *ILLS as hard word sets.
# I thought maybe it would be possible to find a simple combinatorial argument why they can or can't be solved.
# It quickly led to a combinatorial explosion, however . . ..

# Functions to get all words that match i letters of *ESTS or *ILLS, taking into account that to help with 'tests', the word must have two ts or a t in the first position. 
def match_ests(w, l): # So guessing 't' in any but the first position is useless as it'll be yellow in all *ests unless there are two ts
    match_count = set(w) & l
    num_ts = Counter(w)['t']
    if num_ts == 1 and w[0] != 't':
        return match_count - {'t'}
    return match_count

def gests(l, words=words):
    mw = dict() #Matching Words
    for i in range(5):
        mw[i + 1] = set([w for w in words if len(match_ests(w, l)) > i]) # len(set(w) & l1) > i]
    return mw

def match_gills(w, l): 
    match_count = set(w) & l
    num_ss = Counter(w)['s']
    num_ls = Counter(w)['l']
    if num_ss == 1 and w[0] != 's':
        match_count -= {'s'}
    if num_ls < 3 and w[0] != 'l':
        match_count -= {'l'}
    return match_count

def gills(l):
    rw = dict() #Matching Words
    for i in range(5):
        rw[i + 1] = set([w for w in words if len(match_gills(w, l)) > i]) # len(set(w) & l1) > i]
    return rw

#['bests', 'fests', 'gests', 'hests', 'jests', 'kests', 'lests', 'nests', 'pests', 'rests', 'tests', 'vests', 'wests', 'yests', 'zests']
#{'y', 'j', 'l', 'k', 'g', 'f', 'z', 't', 'r', 'b', 'n', 'h', 'v', 'p', 'w'}
hest = [w for w in words if w[1:] == 'ests'] # len = 15 
#l0 = set([w[0] for w in hest]) # len = 15
#shest = set(hest)
hills = [w for w in words if w[1:] == 'ills']
#shills = set(hills) 
#r0 = set([w[0] for w in hills]) # len = 19
# What if I try it for this? 
#rw = gills(r0)

# Returns hint-buckets and their counts for words over the possible_words
def get_hc(word, possible_words):
    hints = dict()
    counts = Counter()
    for goal in possible_words:
        hint = get_hints3(word, goal)
        hints[hint] = hints.get(hint, []) + [goal]
        counts[hint] += 1
    return hints, counts

# Hard mode guesses.  Only constraints are that greens must match and yellows must be present
# I was initially filtering all words with letters that are greyedout in the hints.  i
# Making hard mode much harder but easier to crunch.
def get_hg(guess, hint, possible_words):
    remaining_guesses = []
    green_constraints = []
    yellow_constraints = []
    for index in range(0,5):
        if hint[index] == 'g':
            green_constraints.append((guess[index], index))
        elif hint[index] == 'y':
            yellow_constraints.append(guess[index])
    for goal in possible_words:
        allowed = True
        for letter, index in green_constraints:
            if goal[index] != letter:
                allowed = False
                break
        if allowed:
            for letter in yellow_constraints:
                if not letter in goal:
                    allowed = False
                    break
        if allowed:
            remaining_guesses.append(goal)
    return remaining_guesses

# The structure of the memory dictionaries.  Sets/lists can be hashed as sorted tuples. 
moves = dict((i, dict()) for i in range(1,6))
smoves = dict((i, dict()) for i in range(1,6))

# Logs to keep track of how many times I used the memory dictionaries.
move_counts = dict((i, 0) for i in range(1,6))
smove_counts = dict((i, 0) for i in range(1,6))

#moves, smoves, unsolved, solved_indices = pickle.load(open("sdata2.pickle", 'rb'))

def recupdate(moves1, moves2):
    for i, v in moves1.items():
        v.update(moves2[i])
def updatemoves(moves2):
    recupdate(moves, moves2)

# The brute force Wordle solverk that tries to solve bwords with gwords as guesses from move=1. 
# It can do hard mode and can be given an upper limit.  Once it solves a branch within the limit, it doesn't search further.
# You can give it a fixed starting move.  The option to check moves 3 away from the end is probably buggy.
# If order is true, then it will search in the greedy order of which words split bwords into buckets 'best', i.e., with the least-bad largest-bucket.
def brute_force_wordle(bwords, gwords, move=1, history=[], hard=False, limit=69, fixed_start=None, fixed_second=None, checkthree=False, order=False, log_depth=5):
    best_score = len(bwords) + move
    solution = dict()
    good_move = None
    rmove = limit - move
    if rmove <= 0: # move >= limit:
        return best_score, (move, 'infinity', ",".join(bwords), history)
    # check if it's already been broken or solved
    if rmove <= 2 or (checkthree and rmove <= 3): #move >= limit - 2:
        key = tuple(sorted(bwords)); 
        if key in moves[rmove]:
            move_counts[rmove] += 1
            return best_score, solution
        if key in smoves[rmove]:
            good_move = smoves[rmove][key]
            smove_counts[rmove] += 1
    if move < log_depth:
        print(f"Move {move} ({history+['*']}) trying to solve {len(bwords)} words, starting with {bwords[:5]}.")
    lwords = len(bwords)
    if fixed_start:
        i = gwords.index(fixed_start)
        to_guess = [fixed_start]; gwords = to_guess + gwords[:i] + gwords[i+1:]
    else:
        # Play a winning move from memory
        if good_move in gwords:
            i = gwords.index(good_move)
            to_guess = [good_move] + gwords[:i] + gwords[i+1:]
        elif order and move >= 2:
            to_guess = gwords
            to_guess = [t[1] for t in sorted([(score_hint_distribution(word, bwords), word) for word in to_guess])]
        else:
            to_guess = gwords
        if not good_move and fixed_second and move == 2:
            i = gwords.index(fixed_second)
            to_guess = [fixed_second] + gwords[:i] + gwords[i+1:]
        gwords = to_guess
    for i, guess in enumerate(to_guess):
        tmp_history = history + [guess]
        hint_dist, counts = get_hc(guess, bwords)
        score = counts.most_common(1)[0][1]
        # If the worst-case bucket is of size 1, that means we can win in the next move for each hint!
        if score == 1:
            for hint, wordlist in hint_dist.items():
                word = wordlist[0]
                if guess == wordlist[0]:
                    solution[hint] = (move, word, tmp_history)
                    best_score = move
                else:
                    solution[hint] = (move+1, word, tmp_history+[word])
                    best_score = move+1
            if rmove <= 2 or (checkthree and rmove <= 3):
                smoves[rmove][key] = guess
            return best_score, solution
        # Otherwise, if the word doens't separate the words at all or we only have one move left, we lose.
        if not fixed_start and (score == lwords or move == limit - 1):
            continue

        # Now that it's not solved and we still have a chance, we need to try to brute force each hint's bucket of words!
        next_scores = dict()
        worst_score = move
        for hint, wordlist in hint_dist.items():
            lwd = len(wordlist)
            if lwd > 1:
                if hard:
                    possible_guesses = get_hg(guess, hint, gwords) 
                else:
                    possible_guesses = gwords[i+1:]
                score, next_score = brute_force_wordle(wordlist, possible_guesses, move=move+1, history=tmp_history, hard=hard, limit=limit, checkthree=checkthree, order=order, fixed_second=fixed_second)
                worst_score = max(worst_score, score)
                next_scores[hint] = next_score
                if worst_score > limit:
                    break
            else:
                word = wordlist[0]
                if guess == wordlist[0]:
                    next_scores[hint] = (move, word, tmp_history)
                else:
                    next_scores[hint] = (move+1, word, tmp_history+[word])
                    worst_score = max(worst_score, move+1)
        if worst_score < best_score:
            best_score = worst_score
            solution = next_scores
            if worst_score <= limit:  
                if rmove <= 2 or (checkthree and rmove <= 3):
                    smoves[rmove][key] = guess
                return best_score, solution
    if rmove <= 2 and best_score > limit:
        if rmove == 2:
            print("%s ~ ~%s" % (best_score, solution))
        moves[rmove][key] = True
    elif best_score > limit:
        print("%s, %s ~ ~%s" % (move, best_score, solution))
        if checkthree and rmove <= 3:
            moves[rmove][key] = True
    return best_score, solution


# Cute, so it can stay.  :- p
print()
print()
print("-- -- --")
print("Solving all words.")
print("-- -- --")
print()
print()
#score, best = brute_force_wordle(words, zwords, move=1, limit=6)
#score, best = brute_force_wordle(zwords, zwords, move=1, limit=6, fixed_start='swift', fixed_second='lover', checkthree=False, order=True)

# An example of one of the hard word sets I was working with.  Heck, why not leave it in xD
#hw3 = ['zills', 'maxes', 'gages', 'pents', 'lamer', 'jefes', 'memes', 'fents', 'namer', 'bases', 'zests', 'mages', 'peeps', 'vives', 'waxes', 'yeses', 'pixes', 'waker', 'loges', 'jells', 'zexes', 'games', 'faxes', 'waxer', 'waffs', 'waler', 'tills', 'bills', 'vises', 'wizes', 'maker', 'tates', 'faffs', 'mills', 'vells', 'vills', 'bayes', 'eases', 'rares', 'sages', 'wills', 'fames', 'zerks', 'dills', 'gills', 'hajes', 'serks', 'babes', 'tawer', 'river', 'faves', 'fanes', 'pills', 'jakes', 'lills', 'yexes', 'mazer', 'kexes', 'nills', 'gazes', 'sates', 'pizes', 'pipes', 'lawer', 'mazes', 'loves', 'wawes', 'fills', 'sazes', 'kills', 'fezes', 'tests', 'seeps', 'sills', 'sakes', 'fests', 'zezes', 'wises', 'zaxes', 'jaker', 'hazes', 'fazes', 'leges', 'yills', 'jives', 'tents', 'fazed', 'mells', 'waqfs', 'pests', 'hills', 'fakes', 'vaxes', 'janes', 'faked', 'taver', 'wiver', 'viver', 'raker', 'rills', 'wakfs', 'laxer', 'leves', 'cills', 'gares', 'wafer', 'jills', 'james']

# Below are the supposedly challenging word sets I gathered to try to break Wordle.
'''
*ECKS, *ESTS, *UMPS, *EALS, *OOKS, *OCKS, *ELLS, *ARKS, *ERKS, *ENTS, *ERRY, *OLLY, *IGHT, *AILS, A**FS, A**ES, A**ED, A**AS, O**ED, O**ES, A**AR, I**ES, I**ER, 
>>> ww[1] = [w for w in words if w[1:] == 'ecks']                                                         
>>> ww[2] = [w for w in words if w[1:] == 'ests']
>>> ww[3] = [w for w in words if w[1] == 'a' and w[3:] == 'fs']                                           
>>> ww[4] = [w for w in words if w[1:] == 'umps']                                                         
>>> ww[5] = [w for w in words if w[1] == 'a' and w[3:] == 'es']                                           
>>> ww[6] = [w for w in words if w[1] == 'a' and w[3:] == 'ed']                                           
>>> ww[7] = [w for w in words if w[1:]== 'ills']                                                          
>>> ww[8] = [w for w in words if w[1:]== 'eals']                                                          
>>> ww[9] = [w for w in words if w[1] == 'o' and w[3:] == 'ed']                                           
>>> ww[10] = [w for w in words if w[1] == 'o' and w[3:] == 'es']                                          
>>> ww[11] = [w for w in words if w[1:]== 'ooks']                                                         
>>> ww[12] = [w for w in words if w[1:]== 'ocks']                                                         
>>> ww[13] = [w for w in words if w[1:]== 'ells']                                                         
>>> ww[14] = [w for w in words if w[1:]== 'arks']                                                         
>>> ww[15] = [w for w in words if w[1] == 'a' and w[3:] == 'ar']                                          
>>> ww[16] = [w for w in words if w[1:]== 'erks']                                                         
>>> ww[17] = [w for w in words if w[1] == 'i' and w[3:] == 'es']                                          
>>> ww[18] = [w for w in words if w[1] == 'i' and w[3:] == 'er']                                          
>>> ww[19] = [w for w in words if w[1:]== 'ents']                                                         
>>> ww[20] = [w for w in words if w[1] == 'a' and w[3:] == 'as']                                          
>>> ww[22] = [w for w in words if w[1:]== 'erry']                                                         
>>> ww[23] = [w for w in words if w[1:]== 'olly']                                                         
>>> ww[24] = [w for w in words if w[1:]== 'elly']                                                         
>>> ww[25] = [w for w in words if w[1:]== 'ight']                                                         
>>> ww[26] = [w for w in words if w[1:]== 'ails']
>>> ww[27] = [w for w in words if w[1:3] == 'il' and w[4] == 's']
>>> ww[28] = [w for w in words if w[3:] == 'es'] 
'''

#ww, diffs = pickle.load(open("word_difficulties3.pickle", "rb")) 

# For a given word, order the had word set indexes based on how much the word can split them
# I don't fully remember why I decided to place the easiest first.  I think initially it was the other way around and I changed my mind /(^3^)\
# Optionally, always place 7 := *ILLS first.
def get_ww_diff(word, ww, s=None):
    sw = []
    for i, wl in ww.items():
        if s:
            if s == i:
                continue
            wl = list(set(ww[s] + wl))
        l = len(wl)
        score = dscore_hint_distribution(word, wl)
        sw.append( (score / l, score, i) )
    if s:
        wl = ww[s]; l = len(wl); score = dscore_hint_distribution(word, wl)
        return [(score / l, score, s)] + sorted(sw) 
    return sorted(sw)

# Generate the difficulty dictionary.
if False:
    diffs = dict()
    for word in words:
        diffs[word] = get_ww_diff(word, ww, 7)
    pickle.dump( (ww, diffs), open("word_difficulties3.pickle", 'wb'))

# Update the unsolved and solved word dictionaries.
def updateunsolved(unsolved2):
    for word, info in unsolved2.items():
        if not word in unsolved: 
            unsolved[word] = info

def updateindices(solved_indices2):
    for word, indices in solved_indices2.items():
        if word in solved_indices: 
            if len(solved_indices[word]) < len(indices):
                solved_indices[word] = indices
        else:
            solved_indices[word] = indices

# To facilitate updating the memory banks that are shared among a few procesess... Manually as dealing with read/write conflicts seemed annoying.
def dumpdata():
    pickle.dump((moves, smoves, unsolved, solved_indices), open("sdata2.pickle", 'wb'))

def saveprogress():
    moves2, smoves2, unsolved2, solved_indices2 = pickle.load(open("sdata2.pickle", 'rb'))
    updatemoves(moves2)
    updatemoves(smoves2)
    updateindices(solved_indices2)
    updateunsolved(unsolved2)
    dumpdata()

# Splitting the words up to parallelize the lazy way.
# Also throwing in words that worst/best split the data to 'learn from' first.
#>>> len(words) / 4
#3243.0
#inc = int(len(words) / 4)
#words1 = words[:inc]; del words1[words1.index('puppy')]; words1.insert(0, 'puppy'); del words1[words1.index('aider')]; words1.insert(1, 'aider'); del words1[words1.index('crane')]; words1.insert(0, 'crane')
#words2 = words[inc:2*inc]; del words2[words2.index('fuffy')]; words2.insert(0, 'fuffy'); del words2[words2.index('deair')]; words2.insert(1, 'deair')
#words3 = words[2*inc:3*inc]; del words3[words3.index('nunny')]; words3.insert(0, 'nunny'); del words3[words3.index('oared')]; words3.insert(0, 'oared')
#words4 = words[3*inc:4*inc]; del words4[words4.index('yuppy')]; words4.insert(0, 'yuppy'); del words4[words4.index('redia')]; words4.insert(1, 'redia'); del words4[words4.index('salet')]; words4.insert(0, 'salet');

# Code to try and collect smallish word sets that break each word (which ultimately discovered the winning word: 'swift').
if False:
    solved = dict()
    #ww.update(rww) 
    lww = len(ww)
    for word in words1:
        score, best = None, None
        if word in unsolved:
            continue
        bwords = set()
        indices = list()
        broken = False
        tryHeuristic = False#True
        hard = True
        num = 0
        wdiff = diffs[word]
        solved_indices[word] = solved_indices.get(word, [])
        for i in [i for score_avg, score_count, i in wdiff]:
            num += 1
            print(f"++ Starting index {num}/{lww} of {word}. ({indices})")
            ws = set(ww[i])
            indices.append(i); 
            bwords.update(ww[i])
            if i in solved_indices[word]:
                continue
            count = 0
            # This is a relic from before I combined the greedy search with the brute-force solver
            if tryHeuristic:
                best = compute_all_wordles(word, bwords, move=1, limit=6)
                if check_results(best):
                    solved_indices[word].append(i)
                    print(f"- solved by greedy strategy.")
                    continue
                tryHeuristic = False
            sbwords = sorted(bwords); lbw = len(sbwords)
            score, best = brute_force_wordle(sbwords, zwords, move=1, limit=6, fixed_start=word, hard=hard, checkthree=(lbw < 300), order=True)
            if score > 6:
                broken = True
                unsolved_words = bwords - set([r[1] for r in retrieve_all_results(best)])
                unsolved[word] = (score, indices, unsolved_words, best)
                break
            else:
                solved_indices[word].append(i)
                print(f"- The score is {score}.")
        if not broken:
            solved[word] = (score, indices, best)
        print(f"*** {word} is {'NOT' if not broken else ''} broken!")


###
# Actual solutions and some Wordle Assistant classes that can be played with.
###

# Four deterministic solution dictionaries in the 'old' format.  Includes word histories and solution depth info.
official_solution_full = pickle.load(open("solutions/official_goal_solution.pickle", "rb"))
official_solution_hard_full = pickle.load(open("solutions/official_goal_solution_hard.pickle", "rb"))
swift_solution_full = pickle.load(open("solutions/swiftlover6.pickle", 'rb')) # Almost always plays lover second
swift_solution2_full = pickle.load(open("solutions/swiftlover6_2.pickle", 'rb')) # A bit more diversified

# Solution skeletons that just record which move to take for which hint.  Probably recommended.
official_solution = ast.literal_eval(open('solutions/official_goal_solution.pp.txt', 'r').read())
official_solution_hard = ast.literal_eval(open('solutions/official_goal_solution_hard.pp.txt', 'r').read())
swift_solution = ast.literal_eval(open('solutions/swiftlover6.pp.txt', 'r').read())
swift_solution2 = ast.literal_eval(open('solutions/swiftlover62.pp.txt', 'r').read())


# Wordle assistant that tells you how to play according to a solution file.
# Only works with the 'old' format.
class OWordle:
    def __init__(self, dic=None, move=None):
        self.dic = dic
        if not move:
            move = retrieve_first_result(self.dic)[0][0]
        self.history = [move]
        print(f"Please play '{move}' for move {len(self.history)}.")

    def move(self, hint):
        if isinstance(self.dic, tuple):
            print(f"The game is over.")
            return self.dic[0][-1]
        self.dic = self.dic[hint]
        move_num = len(self.history)
        next_move = retrieve_first_result(self.dic)[0][move_num]
        self.history.append(next_move)
        if isinstance(self.dic, tuple):
            print(f"You have won with '{next_move}' in {len(self.history)} moves.")
        else:
            print(f"Please play '{next_move}' for move {len(self.history)}.")
        return next_move

# Wordle assistant that tells you how to play according to a skeleton of a solution file.
class SWordle:
    def __init__(self, dic=None):
        move = dic[0]
        self.dic = dic[1]
        self.history = [move]
        print(f"Please play '{move}' for move {len(self.history)}.")

    def move(self, hint):
        if isinstance(self.dic, str):
            print(f"The game is over.")
            return self.dic
        move_num = len(self.history) + 1
        next_pair = self.dic[hint]
        if isinstance(next_pair, str):
            next_move = next_pair
            self.dic = None
            print(f"You have won with '{next_move}' in {move_num} moves.")
        else:
            next_move = self.dic[hint][0]
            self.dic = self.dic[hint][1]
            print(f"Please play '{next_move}' for move {move_num}.")
        self.history.append(next_move)
        return next_move

# Wordle assistant to keep track of the possible goals and move history
# It recommends 13 moves each round and you can inspect the whole ranking.
# If you use it's recommendation, you just need to supply the hint.
class GWordle:
    def rec(self, hint, move=None):
        self.move += 1
        if not move:
            move = self.next_move
        self.words = get_hint_distribution_with_words(move, self.words)[hint] 
        #self.words = get_hint_distribution2_with_words(move, self.words)[hint] 
        if len(self.words) == 1:
            next_move = list(self.words)[0]
            print(f"You will win with '{next_move}' in {self.move + 1} moves.")
            return next_move 
        else:
            self.options = sorted([(score_hint_distribution(word, self.words), word) for word in self.guesses]) 
            #options = sorted([(score_hint_distribution2_with_words(word, self.words), word) for word in self.guesses]) 
            to_show = len(self.options) if len(self.options) < 13 else 13
            print(f"There are {len(self.words)} possible words.  The best {to_show} options are:\n{self.options[:to_show]}.")
            self.next_move = self.options[0][1]
            return self.next_move

    # Generates the list of scores for the first move.  May take a while.
    def init_options(self):
        if not self.options:
            self.options = sorted([(score_hint_distribution(word, self.words), word) for word in self.guesses])

    def get_options_dic(self):
        return dict( (word, score) for score, word in self.options )

    def __init__(self, move=None, hint=None, possible_words=words, possible_guesses=words):
        self.words = possible_words
        self.guesses = possible_guesses
        self.options = None
        self.move = 0
        self.next_move = move if move else 'salet'
        if hint:
            rec(hint)

#x = GWordle()
xo = GWordle(possible_words=official_goals)
#s = SWordle(swift_solution2)
#ho = OWordle(official_solution_hard)


# Uses the Wordle solver class to simulate gameplay for each goal
# This is a sanity-check to make sure my solution is correct.
if False:
    sol = pickle.load(open("swiftlover6.pickle", 'rb'))
    fails = 0
    for n, goal in enumerate(words):#official_goals):
        print(f"Word {n} -- {goal}:")
        word = 'swift'
        wordle_instance = OWordle(dic=sol, move=word)
        i = 1
        while word != goal and i < 7:
            h = get_hints3(word, goal)
            word = wordle_instance.move(h)
            i += 1
        if i > 6:
            print(f"FAILED!")
            fails += 1
    print(f"\n\nFailed {fails} times.") 






# I'll end the file with present Zar's mettic wish.

#Zetta :- 
#    In an easy and relaxed manner, 
#    in a healthy and positive way, 
#    in its own perfect time, 
#    for the highest good of all, 
#    and with divine balance, 
#    if you allow it, 
#        I wish you the ongoing fulfillment of your needs, desires, dreams and fantasies, 
#        possibly forms you haven't asked for or imagined.  
#    If you do not allow it, 
#        I wish this energy freely go wherever it is welcome and appreciated.

# <3



