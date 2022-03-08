from collections import Counter
from functools import reduce
from operator import itemgetter
import numpy as np
from pathlib import Path
from random import sample, random
import pickle
import json
import time

# The 12,972 words
words = open("words.txt", 'r').read().splitlines()
wordset = set(words)

# The official words
official_goals = open("wordle_answers_alphabetical.txt", 'r').read().splitlines()
official_goal_set = set(official_goals)

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
                w1 = l2[0][1:].
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

# First stab at ranking words by how well they split the goals
#word_rankings = pickle.load(open("work_rankings_1.pickle", 'rb'))
#ranked_words = list(list(zip(*sorted(word_rankings)))[1])

# WAY faster than locate_word_possibilities!
def get_hint_distribution(word, possible_words):
    hints = Counter()
    for goal in possible_words:
        hint = get_hints3(word, goal)
        hints[hint] += 1
    return hints

def score_hint_distribution(word, possible_words):
    return get_hint_distribution(word, possible_words).most_common(1)[0][1]

def dscore_hint_distribution(word, possible_words):
    hints = get_hint_distribution(word, possible_words)
    total = sum(hints.values())
    acc = 0
    for hint, count in hints.items():
        score = dscore(hint)
        acc += score * (count**2)
    return acc / total

def get_hint_distribution_with_words(word, possible_words):
    hints = dict()
    for goal in possible_words:
        hint = get_hints3(word, goal)
        hints[hint] = hints.get(hint, []) + [goal]
    return hints

def get_hint_distribution_with_words_as_sets(word, possible_words):
    hints = dict()
    for goal in possible_words:
        hint = get_hints3(word, goal)
        tmp = hints.get(hint, set()); tmp.update([goal])
        hints[hint] = tmp
    return hints

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

def count_pair_distribution_with_words(word1, word2, possible_words):
    hints = get_pair_distribution_with_words(word1, word2, possible_words)
    counts = Counter()
    for h, s in hints.items():
        counts[h] = len(s)
    return counts

def score_pair_distribution(word1, word2, possible_words):
    counts = count_pair_distribution_with_words(word1, word2, possible_words)
    return counts.most_common(1)[0][1]

#c = get_hint_distribution(word, words)
#d = get_hint_distribution_with_words(word, words)
#word_distros[word] = (c, d)
#word_distros = pickle.load(open("word_distros.pickle", "rb"))
best_1k, wc_counts = pickle.load(open("word_rankings_3.pickle", "rb"))
worst_words = list(wordset - set(best_1k))

def score_hint_distribution2(word):
    return word_distros[word][0].most_common(1)[0][1]

def get_hint_distribution2(word):
    return word_distros[word][1]

def get_pair_distribution2(word1, word2):
    hint_dist1 = get_hint_distribution2(word1)
    hint_dist2 = get_hint_distribution2(word2)
    hints = dict()
    for hint1, ws1 in hint_dist1.items():
        for hint2, ws2 in hint_dist2.items():
            bucket = set(ws1) & set(ws2)
            if bucket:
                nhint = hint1 + hint2
                hints[nhint] = bucket 
    return hints

def count_pair_distribution2(word1, word2):
    hints = get_pair_distribution2(word1, word2)
    counts = Counter()
    for h, s in hints.items():
        counts[h] = len(s)
    return counts

def score_pair_distribution2(word1, word2):
    counts = count_pair_distribution2(word1, word2)
    return counts.most_common(1)[0][1]


def get_pair_distribution2_with_words2(word1, word2, possible_words):
    hint_dist1 = get_hint_distribution2(word1)
    hint_dist2 = get_hint_distribution2(word2)
    hints = dict()
    for goal in possible_words:
        hint1 = get_hints3(word1, goal); hint2 = get_hints3(word2, goal); nhint = hint1+hint2
        tmp = hints.get(nhint, set()); tmp.update(set(hint_dist1[hint1]) & set(hint_dist2[hint2]))
        hints[nhint] = tmp
    return hints

def get_pair_distribution2_with_words(word1, word2, possible_words):
    hint_dist1 = get_hint_distribution2(word1)
    hint_dist2 = get_hint_distribution2(word2)
    hints = dict()
    ws = set(possible_words)
    for hint1, ws1 in hint_dist1.items():
        for hint2, ws2 in hint_dist2.items():
            bucket = set(ws1) & set(ws2) & ws
            if bucket:
                nhint = hint1 + hint2
                hints[nhint] = bucket 
    return hints

def get_hint_distribution2_with_words(word, possible_words):
    hint_dist = get_hint_distribution2(word)
    pws = set(possible_words)
    hints = dict()
    for hint, ws in hint_dist.items():
        bucket = set(ws) & pws
        if bucket:
            hints[hint] = bucket
    return hints

def score_hint_distribution2_with_words(word, possible_words):
    hint_dist = get_hint_distribution2(word)
    pws = set(possible_words)
    worst = 0
    for hint, ws in hint_dist.items():
        size = len(set(ws) & pws)
        if size > worst :
            worst = size
    return worst

def score_pair_distribution2_with_words2(word1, word2, possible_words):
    hint_dist1 = get_hint_distribution2(word1)
    hint_dist2 = get_hint_distribution2(word2)
    ws = set(possible_words)
    worst = 0
    for hint1, ws1 in hint_dist1.items():
        for hint2, ws2 in hint_dist2.items():
            size = len(set(ws1) & set(ws2) & ws)
            if size > worst:
                worst = size
    return worst

# = time.time(); pd3 = get_pair_distribution2_with_words2('salet', 'irony', words); print(f"{time.time() - s}")
# 1.440363883972168
# >>> s = time.time(); pd = get_pair_distribution2_with_words('salet', 'irony', words); print(f"{time.time() - s}")
# 0.3057122230529785


def count_pair_distribution2_with_words(word1, word2, possible_words):
    hints = get_pair_distribution2_with_words(word1, word2, possible_words)
    counts = Counter()
    for h, s in hints.items():
        counts[h] = len(s)
    return counts

def score_pair_distribution2_with_words(word1, word2, possible_words):
    counts = count_pair_distribution2_with_words(word1, word2, possible_words)
    return counts.most_common(1)[0][1]

# Not sure the desired behavior.  When you can't play the second move...?
def get_pair_distribution2_hardmode(word1, word2):
    hint_dist1 = get_hint_distribution2(word1)
    hint_dist2 = get_hint_distribution2(word2)
    hints = dict()
    for hint1, ws1 in hint_dist1.items():
        if word2 in ws1:
            for hint2, ws2 in hint_dist2.items():
                nhint = hint1 + hint2
                hints[nhint] = set(ws1) & set(ws2)
        else:
            hints[hint1] = set()
    return hints

if False:
    pair_distro = {}
    best = num_words
    best_pair = ("fuck", "you")
    for i1 in range(1000):
        print(f"Beginning word #{i1}.  Current best is {best_pair} at score {best}. ")
        for i2 in range(i1+1, 1000):
            word1 = best_1k[i1]
            word2 = best_1k[i2]
            score = score_pair_distribution(word1, word2, words)
            pair_distro[(word1, word2)] = score
            if score < best:
                best = score
                best_pair = (word1, word2)
                print(f"*** New best: {best_pair} at score {best}. ***")
            print(f"{word1}, {word2} worst case bucket is {score}")
    print(f"Beginning the worse words.")
    for word1 in best_1k:
        for word2 in worst_words:
            score = score_pair_distribution(word1, word2, words)
            pair_distro[(word1), (word2)] = score
            # Best serai: [(174, 'serai', 'loans'), (189, 'serai', 'loast'), (189, 'serai', 'lotsa'),
            print(f"{word1}, {word2} worst case bucket is {score}")

if False:
    salet_pair_distro = {}
    hint_groups = get_hint_distribution_with_words('salet', words)
    for hint, wordlist in hint_groups.items():
        l = len(wordlist)
        pair_list = []
        best = num_words
        for i1 in range(1000):
            for i2 in range(i1+1, 1000):
                word1 = best_1k[i1]
                word2 = best_1k[i2]
                score = score_pair_distribution(word1, word2, wordlist)
                pair_list.append((score, ((word1), (word2))))
                if score < best:
                    best = score
                    print(f"For {hint} of size {l}, {word1}, {word2} worst case bucket is {score}")
        print(f"Hint {hint} can be reduced from {l} to {best}.")
        pair_list = sorted(pair_list)
        salet_pair_distro[hint] = pair_list[:100]

# pickle.dump(pair_distro, open("pair_distro_parts_1-68.pickle", "wb"))
#[(129, 'salet', 'irons'), (132, 'salet', 'noirs'), (132, 'salet', 'noris'), (132, 'solei', 'rants'), (132, 'solei', 'tarns'), (134, 'nates', 'reoil'), (134, 'tales', 'irons'), (136, 'setal', 'irons'), (136, 'setal', 'noris'), (144, 'nates', 'loirs'), (144, 'nates', 'loris'), (144, 'nates', 'roils'), (144, 'tales', 'noirs'), (144, 'tales', 'noris'), (145, 'canes', 'reoil'), (147, 'cates', 'irons'), (147, 'nates', 'soral'), (147, 'rates', 'sloan'), (147, 'rates', 'solan'), (147, 'setal', 'noirs'), (147, 'stoae', 'nirls'), (147, 'taces', 'irons'), (147, 'tares', 'sloan'), (147, 'tares', 'solan'), (149, 'cates', 'reoil'), (149, 'taces', 'reoil'), (152, 'canes', 'loirs'), (152, 'canes', 'loris'), (152, 'canes', 'roils'), (152, 'reais', 'sloan'), (152, 'solei', 'darts'), (155, 'slate', 'irons'), (155, 'slate', 'noirs')]
# Best serai: [(174, 'serai', 'loans'), (189, 'serai', 'loast'), (189, 'serai', 'lotsa'),


#pickle.dump(salet_pair_distro, open("salet_pair_distro.pickle", 'wb'))
#salet_pair_distro = pickle.load(open("salet_pair_distro.pickle", 'rb'))

#>>> pickle.dump(sol, open("tmp_hardwords.pickle", "wb"))
#>>> hardwords
#['balls', 'bells', 'bills', 'bolls', 'bulls', 'calls', 'cells', 'cills', 'colls', 'culls', 'dells', 'dills', 'dolls', 'dulls', 'falls', 'fells', 'fills', 'fulls', 'galls', 'gills', 'gulls', 'halls', 'hells', 'hills', 'hulls', 'jells', 'jills', 'jolls', 'kells', 'kills', 'lalls', 'lills', 'lolls', 'lulls', 'malls', 'mells', 'mills', 'molls', 'mulls', 'nills', 'nolls', 'nulls', 'palls', 'pells', 'pills', 'polls', 'pulls', 'rills', 'rolls', 'sells', 'sills', 'talls', 'tells', 'tills', 'tolls', 'vells', 'vills', 'walls', 'wells', 'wills', 'wulls', 'yells', 'yills', 'zills', 'bests', 'fests', 'gests', 'hests', 'jests', 'kests', 'lests', 'nests', 'pests', 'rests', 'tests', 'vests', 'wests', 'yests', 'zests']
#hardsol = pickle.load(open("tmp_hardwords.pickle", "rb"))

#!!! Aha!  
#>>> score_pair_distribution('salet', 'irony', words)
#127
#>>> score_pair_distribution('salet', 'irons', words)
#129

# So salet, irons is the best least-bad pair.  Now I'd wanna find the best pair assuming salet is first?
# Move 1 == salet.
# So this process would deal with Move 2 and Move 3... or at least Move 2.
# As while salet -> irons is the best case for its bucket, ybbbb, maybe there are better choices than irons for the other buckets! 

#Counter({'bbbbb': 865, 'ybbbb': 824, 'bbbyb': 715, 'bybbb': 583, 'bbbgb': 552, 'bgbbb': 441, 'ybbyb': 401, 'ygbbb': 393, 'ybbgb': 364, 'yybbb': 340, 'bybyb': 301, 'gbbbb': 269, 'bbbby': 269, 'bbybb': 263, 'byybb': 256, 'ybbby': 256, 'bbyyb': 208, 'bgbgb': 198, 'bbbyy': 195, 'ybybb': 173, 'gybbb': 167, 'gbbyb': 151, 'bbygb': 147, 'gbbgb': 129, 'byyyb': 129, 'ybbyy': 126, 'bybby': 125, 'ygbby': 123, 'bbbbg': 119, 'ygbgb': 119, 'bgbby': 118, 'bbbgy': 113, 'yybyb': 111, 'bgbyb': 101, 'bbgbb': 100, 'yyybb': 99, 'ybgbb': 98, 'gbbby': 95, 'bgybb': 92, 'ybyyb': 87, 'yybby': 85, 'bybyy': 81, 'bybbg': 81, 'gbybb': 73, 'ggbbb': 73, 'ybbgy': 72, 'bygbb': 68, 'gbbyy': 64, 'ybygb': 64, 'ygybb': 63, 'bbbyg': 62, 'gybyb': 60, 'bybgb': 58, 'bbgyb': 57, 'bggbb': 57, 'bbggb': 56, 'ybbbg': 53, 'gybby': 52, 'gyybb': 48, 'bbbgg': 44, 'gbyyb': 44, 'gbbbg': 43, 'yggbb': 41, 'bgbbg': 37, 'bgbgy': 37, 'bbyyy': 37, 'ybggb': 37, 'bbyby': 36, 'yybgb': 35, 'ybgyb': 35, 'bgygb': 34, 'yyyyb': 34, 'yybyy': 33, 'ggbgb': 32, 'byyby': 31, 'bbybg': 30, 'ybbyg': 30, 'bgyyb': 30, 'bybyg': 27, 'gbgbb': 27, 'ybyby': 25, 'ygbyb': 23, 'ygbgy': 23, 'ybgby': 22, 'bgbyy': 20, 'gybbg': 19, 'byybg': 19, 'gybyy': 19, 'bgyby': 18, 'yygbb': 18, 'yybbg': 16, 'bygyb': 16, 'gbygb': 16, 'gbbgy': 15, 'bbgby': 15, 'gyyyb': 14, 'bggby': 14, 'bgggb': 14, 'gbyby': 13, 'gggbb': 13, 'bbyyg': 13, 'bbygy': 13, 'ygygb': 13, 'byygb': 12, 'ygggb': 12, 'ybgyy': 12, 'byyyy': 11, 'byggb': 11, 'gbggb': 11, 'bbgyy': 10, 'ygbyy': 10, 'ggbby': 10, 'byyyg': 9, 'ybbgg': 9, 'bbggg': 9, 'gbbyg': 9, 'yyyby': 9, 'gygbb': 8, 'ggbyb': 8, 'yggby': 8, 'ygbbg': 7, 'yybyg': 7, 'bybgy': 7, 'bgbgg': 7, 'ybyyy': 7, 'bgybg': 6, 'bggyb': 6, 'gbybg':6, 'gbyyy': 6, 'bbgyg': 6, 'ygyby': 6, 'gbgyb': 5, 'ygyyb': 5, 'ggbbg': 5, 'gyyyy': 5, 'yyybg': 5, 'ggybb': 5, 'bgygy': 5, 'gyybg': 5, 'bbggy': 5, 'bygby': 5, 'bygyy': 4, 'bbgbg': 4, 'gbyyg': 4, 'gyyby': 4, 'bbygg': 4, 'bgyyy': 4, 'gggyb': 4, 'yyygb': 4, 'ybygy': 4, 'ybggy': 4, 'gbbgg': 3, 'gybyg': 3, 'yggyb': 3, 'byygy': 3, 'gggby': 3, 'yyggb': 3, 'yybgy': 3, 'yyyyy': 3, 'ggbgy': 3, 'gygyb': 3, 'gybgb': 3, 'yybgg': 2, 'bgggg': 2, 'byggg': 2, 'bybgg': 2, 'yygby': 2, 'ybybg': 2, 'yygyb': 2, 'gggbg': 2, 'ggggb': 2, 'gbgby': 2, 'bgggy': 2, 'ybggg': 1, 'ggbyy': 1, 'bygyg': 1, 'gbygg': 1, 'gygbg': 1, 'gbygy': 1, 'yyyyg': 1, 'gbgbg': 1, 'bygbg': 1, 'ybyyg': 1, 'bgbyg': 1, 'ygbyg': 1, 'bggbg': 1, 'bgygg': 1, 'ggyyb': 1, 'ggggg': 1, 'ggygb': 1, 'ggybg': 1, 'gyygb': 1, 'ygyyy': 1, 'ygggy': 1})

'''
for i in range(0, num_words):
    word = words[i]
    for j in range(i, num_words):
        word2 = words[j]
        word_ranks.append( (score_pair_distribution(word, word2, sd['ybgbb']), word, word2) ) 

# omfg, the process was killed AS I WAS DUMPING IT
pickle.dump((word_ranks, i, j-1, sd['ybgbb']), open("word_ranks_ybgbb_part1.pickle", 'wb'))

#>>> i
#9566
#>>> j
#11253
#>>> len(word_ranks)
#78342444
'''

h1 = list(filter(lambda w : w[2:] == "lls", words))
h2 = list(filter(lambda w : w[1:] == "ests", words))
hardwords = h1+h2
count = 0

# So in each step I want to receive a word and a list of valid words (initially seeded)
# Then I compute a dictionary yof all possible hints.
# Next I discharge the solution of all buckets.
def compute_all_wordles(bword, bwords, move=1, extra=False, history=[], limit=69):
    global count
    if move >= limit:
        print(f"Limit {limit} exceeded with {bword}:{bwords[:5]}.")
        return (history + bwords, move + len(bwords))
    #goals = dict()
    solutions = dict()
    #for word in bwords:
    #    hint = get_hints3(bword, word)
    #    goals[hint] = goals.get(hint, []) + [word]
    goals = get_hint_distribution_with_words(bword, bwords)
    #next_hint = sorted([(hint, len(wordlist)) for hint, wordlist in goals.items()], key=itemgetter(1), reverse=True)[0][0]
    #print(f"There are {len(goals)} distinct hints possible on move {move} with '{bword}'")
    #print(f"{[(hint, wordlist[:1]) for hint, wordlist in goals.items()]}")
    for hint, wordlist in goals.items():
        #if move == 5:
        #    wordlist = list(set(wordlist) & official_goal_set)
        #    if len(wordlist) == 0:
        #        solutions[hint] = (history + [bword] + wordlist, 7)
        #        continue
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
        #next_word = sorted([(word, locate_word_possibilities(word, wordlist)) for word in words], key=itemgetter(1))[0][0]
        elif extra:
            #if len(wordlist) < 50 and move == 3:
            #    next_word = sorted([(score_pair_distribution(word, word2, wordlist), word) for word in words for word2 in (best_1k + wordlist + sample(words, 33))])[0][1]
            #else:
                next_word = sorted([(score_hint_distribution(word, wordlist), word) for word in words])[0][1]
        elif move == 0:
            #next_word = sorted([(score_pair_distribution(word, word2, wordlist), word) for word in (['irony'] + best_1k + wordlist) for word2 in wordlist])[0][1]
            if False: #len(wordlist) > 500:
                #next_word = sorted([(score_pair_distribution2_with_words2(word, word2, wordlist), word) for word in words for word2 in ['irony'] + best_1k + wordlist])[0][1]
                next_word = sorted([(score_pair_distribution2_with_words2(word, word2, wordlist), word) for word in (['irony'] + best_1k + wordlist) for word2 in wordlist])[0][1]
            else:
                #next_word = sorted([(score_pair_distribution(word, word2, wordlist), word) for word in words for word2 in words])[0][1]
                next_word = sorted([(score_pair_distribution(word, word2, wordlist), word) for word in (['irony'] + best_1k + wordlist) for word2 in wordlist])[0][1]
            #next_word = sorted([(score_pair_distribution(word, word2, wordlist), word) for word in words for word2 in wordlist])[0][1]
            #next_word = sorted([(get_hint_distribution(word, wordlist).most_common(1)[0][1], word) for word in words])[0][1]
        else:
            if False: #len(wordlist) > 500:
                next_word = sorted([(score_hint_distribution2_with_words(word, wordlist), word) for word in words])[0][1]
            else:
                #next_word = sorted([(get_hint_distribution(word, wordlist).most_common(1)[0][1], word) for word in words])[0][1]
                next_word = sorted([(score_hint_distribution(word, wordlist), word) for word in words])[0][1]
        solutions[hint] = compute_all_wordles(next_word, wordlist, move=move+1, history=(history+[bword]), limit=limit)
        # Oh, lol, I could check-solutions, here, and try all next words... >:D.
        # Or at least, try maybe 10 next words before giving up and going to the next level >:D
    return solutions

# Set up: bwords = official and guesses = full list, so it should always include and be bigger than the official.
def compute_all_wordles_hard(bword, bwords, guesses=None, move=1, extra=False, lookahead=1, history=[]):
    solutions = dict()
    if not guesses:
        guesses = bwords
    goals = get_hint_distribution_with_words(bword, bwords)
    guessdic = get_hint_distribution_with_words(bword, guesses)
    for hint, wordlist in goals.items():
        guesslist = guessdic[hint]
        if len(wordlist) == 1:
            if bword == wordlist[0]:
                solutions[hint] = (history + [bword], move)
                if move > 6:
                    print(f"Solved {hint}:{wordlist[0]} on move {move} with guesses: {history + [bword]}.")
            else:
                solutions[hint] = (history + [bword] + wordlist, move+1)
                if move > 5:
                    print(f"Solved {hint}:{wordlist[0]} on move {move+1} with guesses: {history + [bword]}.")
            continue
        if move <= lookahead:
            #next_word = sorted([(score_pair_distribution(word, word2, wordlist), word) for word in wordlist for word2 in wordlist])[0][1]
            next_word = sorted([(score_pair_distribution(word, word2, wordlist), word) for word in guesslist for word2 in guesslist])[0][1]
        elif extra:
            #next_word = sorted([(score_hint_distribution(word, wordlist), word) for word in wordlist])[0][1]
            next_word = sorted([(score_hint_distribution(word, wordlist), word) for word in guesslist])[0][1]
        else:
            #next_word = sorted([(get_hint_distribution(word, wordlist).most_common(1)[0][1], word) for word in wordlist])[0][1]
            next_word = sorted([(get_hint_distribution(word, wordlist).most_common(1)[0][1], word) for word in guesslist])[0][1]
        solutions[hint] = compute_all_wordles_hard(next_word, wordlist, guesses=guesslist, move=move+1, lookahead=lookahead, history=(history+[bword]))
    return solutions


#>>> s = time.time(); d = get_hint_distribution_with_words('aeros', words); e = time.time(); print(f"{e - s}")
#0.03649020195007324
#>>> s = time.time(); d1 = get_hint_distribution('aeros', words); e = time.time(); print(f"{e - s}")
#0.02434992790222168
# vs like 25-30 seconds for the other one!
'''
word_distros = dict()
for i, word in enumerate(words):
  if i % 1000 == 0:
      print(f"On step {i} / {num_words}")
  c = get_hint_distribution(word, words)
  d = get_hint_distribution_with_words_as_sets(word, words)
  word_distros[word] = (c, d)
pickle.dump(word_distros, open("word_distros2.pickle", "wb"))
# 833M!
# Updated in the same file... whoops, but I fixed it.
wc_counts = sorted([(c.most_common(1)[0][1] , word) for word, (c, _) in word_distros.items()])
best_1k2 = [w for _, w in wc_counts][:1000]
#pickle.dump((best_1k2, wc_counts), open("word_rankings_3.pickle", "wb"))
'''
#>>> wc_counts[:10]
#[(697, 'serai'), (769, 'reais'), (769, 'soare'), (776, 'paseo'), (801, 'aeros'), (821, 'kaies'), (823, 'nares'), (823, 'nears'), (823, 'reans'), (825, 'stoae')]
#>>> wc_counts[-1]
#(8189, 'gyppy')
#Interesting... it's mostly the same but a biiiit different
#pickle.dump((best_1k, wc_counts), open("word_rankings_2.pickle", "wb"))
#>>> dscore_distros = {}
#>>> for word in words:
#    ...  dscore_distros[word] = score_hint_distribution(word, words)
#    ... 
#    >>> sc = sorted([(s, w) for w,s in dscore_distros.items()])
#>>> sc[:10]
#[(2375.1298951588037, 'lares'), (2403.1796176379894, 'rales'), (2501.012565525748, 'reais'), (2503.9330866481655, 'tares'), (2510.490749306198, 'nares'), (2511.9708603145236, 'aeros'), (2513.8434320074007, 'soare'), (2575.955673758865, 'rates'), (2603.6278137526983, 'arles'), (2653.5081714461917, 'aloes')]
#>>> sc[-10:]
#[(48949.254239901325, 'jugum'), (49075.940178846744, 'yukky'), (49280.66057662658, 'bubby'), (49350.35800185014, 'cocco'), (50011.36578785076, 'fuzzy'), (51477.88629355535, 'immix'), (51865.8388066605, 'hyphy'), (53230.85892691952, 'xylyl'), (53373.009019426456, 'gyppy'), (53518.9004008634, 'fuffy')]

#p4 = compute_all_wordles('aeros', words, move=1)
#pickle.dump(p4, open("wordle_pseudo_solution1.pickle", "wb"))
#sol_dic = pickle.load(open("wordle_pseudo_solution1.pickle", 'rb'))
#pickle.dump(sol_dic, open("wordle_pseudo_solution2.pickle", "wb")) # failes 57
#pickle.dump(sol_dic2, open("wordle_pseudo_solution3.pickle", "wb")) # failes 45 -- with 'salet' as seed!
#pickle.dump(sol_dic2, open("wordle_pseudo_solution3.pickle", "wb")) # failes 65 -- with 'salet' as seed!
sol_dic = pickle.load(open("wordle_pseudo_solution3.pickle", 'rb'))
sol_dic2 = pickle.load(open("wordle_pseudo_solution5.pickle", 'rb'))

def check_results(solutions):
    if isinstance(solutions, tuple):
        if solutions[1] > 6:
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

def retrieve_false_results(solutions, hints = []):
    results = []
    if isinstance(solutions, tuple):
        if solutions[1] > 6:
            return [(hints, solutions)]
        return []
    for h, v in solutions.items():
        results += retrieve_false_results(v, hints + [h])
    return results

def retrieve_first_result(solutions):
    if isinstance(solutions, tuple):
        return solutions
    return retrieve_first_result(next(iter(solutions.values())))

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

def fixup_history(solutions, history):
    if isinstance(solutions, tuple):
            return (solutions[0], solutions[1], history + solutions[2])
    for hint, next_solutions in solutions.items():
        solutions[hint] = fixup_history(next_solutions, history)
    return solutions

def translate_res_new_to_old(solutions):
    if isinstance(solutions, tuple):
            return (solutions[2], solutions[0])
    for hint, next_solutions in solutions.items():
        solutions[hint] = translate_res_new_to_old(next_solutions)
    return solutions

def translate_res_old_to_new(solutions):
    if isinstance(solutions, tuple):
            return (solutions[1], solutions[0][-1], solutions[0])
    for hint, next_solutions in solutions.items():
        solutions[hint] = translate_res_old_to_new(next_solutions)
    return solutions

def get_sol_entries(d):
    return [wl[-1] for wl, _ in retrieve_all_results(d)]

def find_sol_hint(w, d):
    for key, value in d.items():
        if w in get_sol_entries(value):
            return key

def get_sol_skeleton(d, i=1):
    moves = dict()
    for hint, next_solutions in d.items():
        #print(hint)
        res = retrieve_first_result(next_solutions)[0]
        if hint != "ggggg":#len(res) > i:
            next_move = res[i]
            if isinstance(next_solutions, dict):
                moves[hint] = (next_move, get_sol_skeleton(next_solutions, i+1))
            else:
                moves[hint] = next_move
        else:
            moves[hint] = res[-1]
    return moves

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

def get_history_buckets(history_dic):
    if isinstance(history_dic, list):
        return [(len(history_dic), history_dic)]
    results = []
    for value in history_dic.values():
        results += get_history_buckets(value)
    return results

#>>> [find_sol_hint(w, sis['ybbyy']) for w in fs['ybbyy']]
#['bgbbb', 'bgbbb', 'bgbbb']



#base, si_sols, fails = pickle.load(open("tmp.pickle", 'rb'))
#si_sols, fails = pickle.load(open("tmp2.pickle", 'rb'))
#si_sols3, fails3 = pickle.load(open("si_sols3.pickle", 'rb'))
#si_sols, fails = pickle.load(open("official_goal_solution_hard_tmp1.pickle", 'rb'))

#>>> pickle.dump((si_sols, fails), open("si_sols1.pickle", "wb"))
#>>> set([w[:5] for w in fails.keys()])
#{'ybgbb', 'ybbyy', 'ybbgb'}

#>>> pickle.dump((si_sols, fails), open("si_sols2.pickle", "wb"))
#>>> set([w[:5] for w in fails.keys()])
#{'ygbgb', 'ybgbb', 'bgbgb', 'ybbgb', 'ybbyy', 'bbbgb'}

#>>> pickle.dump((si_sols, fails), open("si_sols3.pickle", "wb"))
#>>> set([w[:5] for w in fails.keys()])
#{'ybgbb', 'ybbyy'}

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

wordlist2 = []
next_words = []

def try_again(fails, partials, base=None, trynum=0, extra=False, official_goals_only=False, history=['salet']):
    global count
    global wordlist2
    global next_words
    for hints in fails.keys():
        if base:
            wordlist = base[hints]
        else:
            wordlist = [w[0][-1] for w in retrieve_all_results(partials[hints])]
            if not extra and set(wordlist) != set(wordlist2):
                wordlist2 = wordlist
                #next_words = sorted([(score_pair_distribution(word, word2, wordlist2), word) for word in words for word2 in words])
                next_words = sorted([(score_hint_distribution2_with_words(word, wordlist), word) for word in words])
        if extra:
            bword = sorted([(dscore_hint_distribution(word, wordlist), word) for word in words])[trynum][1]
        else:
            bword = next_words[trynum][1]
            #bword = sorted([(score_hint_distribution2_with_words(word, wordlist), word) for word in words])[trynum][1]
        count = 0
        partials[hints] = compute_all_wordles(bword, wordlist, move=2, extra=extra, history=history)
        if check_results(partials[hints]):
            print(f"{hints} is newly solved with {bword}.")
    return partials

def try_again_hard_official(fails, partials, guessdic, trynum=0, extra=False):
    for hints in fails.keys():
        wordlist = [w[0][-1] for w in retrieve_all_results(partials[hints])]
        guesslist = guessdic[hints]
        if extra:
            bword = sorted([(score_hint_distribution(word, wordlist), word) for word in guesslist])[trynum][1]
        else:
            bword = sorted([(get_hint_distribution(word, wordlist).most_common(1)[0][1], word) for word in guesslist])[trynum][1]
        partials[hints] = compute_all_wordles_hard(bword, wordlist, guesses=guesslist,  move=2, extra=extra, history=['salet'])
        if check_results(partials[hints]):
            print(f"{hints} is newly solved with {bword}.")
    return partials


if False:
    si_sols = pickle.load(open("salet_tmp_sol_1.pickle", 'rb'))
    history = ['salet', 'genre', 'syphs']
    #si_sols = pickle.load(open("logic1.pickle", 'rb'))
    fails, _ = get_fails(si_sols)
    #si_sols = sol_dic2
    #fails, failed4 = get_fails(si_sols)
    #si_sols = si_sols3; fails = fails3
    #guessdic = get_hint_distribution2('logic')
    for i in range(0, num_words):
        for j in range(2):
            if j == 0:
                si_sols = try_again(fails, si_sols, trynum=i, extra=True, history=history)
                #si_sols = try_again_hard_official(fails, si_sols, guessdic=guessdic, trynum=i, extra=True)
                fails, failed4 = get_fails(si_sols, official_goals_only=False)
                print(f"On try {i}-1 there are {len(fails)} failed hints and {len(failed4)} failed words.")
                if len(fails) == 0:
                    break
            else:
                si_sols = try_again(fails, si_sols, trynum=i, extra=False, history=history)
                #si_sols = try_again_hard_official(fails, si_sols, guessdic=guessdic, trynum=i, extra=False)
                fails, failed4 = get_fails(si_sols, official_goals_only=False)
                print(f"On try {i}-2 there are {len(fails)} failed hints and {len(failed4)} failed words.")
                if len(fails) == 0:
                    break
        #print(f"On try {i}-extra={extra} there are {len(fails)} failed hints and {len(failed4)} failed words.")
        if len(fails) == 0:
            break

#>>> si_sols = sis['ybbyy']['bgbbb']
#>>> for word in words:
#    ...  z = compute_all_wordles(word, fs['ybbyy'], move=2, history=['salet'])
#    ...  fails, failed4 = get_fails(z)
#    ...  if len(fails) == 0:
#        ...   adequate_worsd.append(word)


#official solution 2.3k goals
#sol_dico = pickle.load(open("official_goal_solution.pickle", "rb"))
sol_dico = pickle.load(open("official_goal_solution_fixed.pickle", "rb"))
sol_dicho = pickle.load(open("official_goal_solution_hard.pickle", "rb"))

class OWordle:
    def __init__(self, dic=sol_dico, move='salet'):
        self.dic = dic
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

class GWordle:
    def rec(self, hint, move=None):
        self.move += 1
        if not move:
            move = self.next_move
        #self.words = get_hint_distribution_with_words(move, self.words)[hint] 
        self.words = get_hint_distribution2_with_words(move, self.words)[hint] 
        if len(self.words) == 1:
            next_move = list(self.words)[0]
            print(f"You will win with '{next_move}' in {self.move + 1} moves.")
            return next_move 
        else:
            #options = sorted([(score_hint_distribution(word, self.words), word) for word in self.guesses]) 
            options = sorted([(score_hint_distribution2_with_words(word, self.words), word) for word in self.guesses]) 
            to_show = len(options) if len(options) < 13 else 13
            print(f"There are {len(self.words)} possible words.  The best {to_show} options are:\n{options[:to_show]}.")
            self.next_move = options[0][1]
            return self.next_move

    def __init__(self, move=None, hint=None, possible_words=words, possible_guesses=words):
        self.words = possible_words
        self.guesses = possible_guesses
        self.move = 0
        self.next_move = move if move else 'salet'
        if hint:
            rec(hint)

    

x = GWordle()
xo = GWordle(possible_words=official_goals)
#ho = OWordle(sol_dicho)

#bsol = pickle.load(open("swift_bsol_words.pickle", 'rb'))
if False:
    sol = pickle.load(open("swiftlover6.pickle", 'rb'))
    translate_res_new_to_old(sol)
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

# So for each possible hint, get the words
#>>>test = bword_goals['bbbyg']
# Locate the best, worst, average possibilities over this space of words.
#>>>test_p = [locate_pair_possibilities('aeros', word, test) for word in test]
#>>> test_p.index((53.891352549889135, 2, 101))
#47
#>>> test[47]
#'clons'
#>>> sorted(test_p, key=itemgetter(0))[0]
#(53.891352549889135, 2, 101)
# Take the best one, say, clons
#test = bword_goals['bbbyg'] 
#print(len(test)) # 451
#>>> sum(len(ws) for ws in bword_goals2.values())
#451 options remaining?
#>>> sorted([(h,len(ws)) for h, ws in bword_goals2.items()], key=itemgetter(1), reverse=True)
#[('bbybg', 101), ('bbgbg', 77), ('byybg', 66), ('bbyyg', 40), ('bggbg', 25), ('bygbg', 20), ('ybybg', 15), ('gbybg', 15), ('bbggg', 14), ('bbgyg', 13), ('bbygg', 11), ('gbgbg', 9), ('gggbg', 7), ('gyybg', 5), ('byyyg', 5), ('yyybg', 4), ('ybgbg', 4), ('gbggg', 3), ('gbyyg', 3), ('byygg', 3), ('yggbg', 2), ('gbygg', 2), ('byggg', 2), ('ybyyg', 2), ('ggggg', 1), ('gygbg', 1), ('ybggg', 1)]

#test_p = [locate_word_possibilities(word, test) for word in test]

#>>> test_p.index((50.75609756097561, 2, 41))
#44
#>>> test[44]
#'moits'


#for hint, word_list in bword_goals.items():
#    bword_plays[hint]

good_words = ["lurid", "poise", "roast", "caste", "adore", "death", "gymps", "nymph", "fjord", "vibex", "waltz"
             ,"gucks", "vozhd", "waqfs", "treck", "jumpy", "bling", "brick", "clipt", "kreng", "prick", "pling"
             ,"bemix", "clunk", "grypt", "xylic", "cimex", "brung", "blunk", "kempt", "quick", "verbs", "arose"]

def solve_wordle(seed, goal, possible_words, seed2=None, sample_num=0):
    move = seed
    power = 0
    for i in range(0,5):
        if i == 0:
            #power = calculate_word_possibilities2(move, possible_words)
            move = move
        elif i == 1 or i == 2:
            if seed2 and i == 1:
                next_words = get_different_word(move, hints, words)
                move = seed2
                #power = calculate_word_possibilities2(move, possible_words)
            else:
                if i == 1:
                    next_words = get_different_word(move, hints, words)
                elif i == 2:
                    next_words = get_different_word(move, hints, next_words)
                next_moves = []
                if len(possible_words) > 444 or i == 1:
                    good_sample = sample(next_words, sample_num) + good_words
                    for word1 in words:
                        for word2 in good_sample:
                            if word1 != word2:
                                next_moves.append((score_pair_distribution(word1, word2, possible_words), word1))
                                #next_moves.append((calculate_pair_possibilities2_with_dscore(word1, word2, possible_words), word1))
                elif len(possible_words) < 66:
                    for word in (possible_words + good_words) :
                        next_moves.append((score_hint_distribution(word, possible_words), word))
                        #next_moves.append((calculate_word_possibilities2_with_dscore(word, possible_words), word))
                else:
                    for word in (next_words + good_words):
                        next_moves.append((score_hint_distribution(word, possible_words), word))
                        #next_moves.append((calculate_word_possibilities2_with_dscore(word, possible_words), word))
                power, move = sorted(next_moves)[0]
        else:
            next_moves = []
            if i == 3:
                for word in (possible_words + good_words):
                    next_moves.append((score_hint_distribution(word, possible_words), word))
                    #next_moves.append((calculate_word_possibilities2_with_dscore(word, possible_words), word))
            else:
                for word in possible_words:
                    next_moves.append((score_hint_distribution(word, possible_words), word))
                    #next_moves.append((calculate_word_possibilities2_with_dscore(word, possible_words), word))
            power, move = sorted(next_moves)[0]
        hints = get_hints2(move, goal)
        print(f"Move #{i+1} ({move}) hints are {hints} with score of {len(possible_words)} into {power if power > 0  else '-'}.")
        possible_words = filter_word_by_goal2(move, goal, possible_words)
        print(f"There are {len(possible_words)} options remaining.")
        if len(possible_words) == 1:
            break
        if len(possible_words) < 11:
            print(f"They are: {possible_words}.")
    return possible_words

def solve_wordle_broken(seed, goal, possible_words, seed2=None, sample_num=66):
    move = seed
    power = 0
    best_words = ranked_words[:sample_num]
    good_sample = best_words + good_words
    for i in range(0,5):
        start_time = time.time()
        if i == 0:
            #power = calculate_word_possibilities2(move, possible_words)
            move = move
        elif i == 1 or i == 2:
            if seed2 and i == 1:
                next_words = get_different_word(move, hints, ranked_words)
                move = seed2
                #power = calculate_word_possibilities2(move, possible_words)
            else:
                if i == 1:
                    next_words = get_different_word(move, hints, ranked_words)
                elif i == 2:
                    next_words = get_different_word(move, hints, next_words)
                next_moves = []
                if len(possible_words) > 444 or i == 1:
                    good_sample = good_words + next_words[:sample_num]
                    for word1 in good_sample:
                        for word2 in best_words:
                            if word1 != word2:
                                next_moves.append((calculate_pair_possibilities2_with_dscore(word1, word2, possible_words), word1))
                elif len(possible_words) < 100:
                    for word in (possible_words + good_sample) :
                        next_moves.append((calculate_word_possibilities2_with_dscore(word, possible_words), word))
                else:
                    for word in (next_words + good_words):
                        next_moves.append((calculate_word_possibilities2_with_dscore(word, possible_words), word))
                power, move = sorted(next_moves)[0]
        else:
            next_moves = []
            if i == 3:
                for word in (possible_words + good_sample):
                    next_moves.append((calculate_word_possibilities2_with_dscore(word, possible_words), word))
            else:
                for word in possible_words:
                    next_moves.append((calculate_word_possibilities2_with_dscore(word, possible_words), word))
            power, move = sorted(next_moves)[0]
        hints = get_hints2(move, goal)
        print(f"Move #{i+1} ({move}) hints are {hints} with score of {len(possible_words)} into {power if power > 0  else '-'}.")
        possible_words = filter_word_by_goal2(move, goal, possible_words)
        print(f"There are {len(possible_words)} options remaining.")
        end_time = time.time()
        print(f"Round #{i+1} took {end_time - start_time} seconds.")
        if len(possible_words) == 1:
            break
        if len(possible_words) < 11:
            print(f"They are: {possible_words}.")
    return possible_words

if False:
    failed = []
    failed2 = []
    for i, word in enumerate(words):
        possible_words = solve_wordle('stare', word, words, seed2='lurid')
        if len(possible_words) > 1:
            failed.append((word, possible_words))
            # Try again!  Without the additional seed2 :- )
            possible_words2 = solve_wordle('stare', word, words) 
            if len(possible_words2) > 1:
                print(f"Failed on #{i} with {word} and {len(possible_words)} options left.")
                print(f"Failed again on #{i} with {word} and {len(possible_words2)} options left.")
                print(possible_words)
                print(possible_words2)
                failed2.append((word, possible_words2))
            else:
                print(f"#{i}: {word} is solved on take two.")
        else:
            print(f"#{i}: {word} is solved.")
    print(f"Success rate 1 is {1 - len(failed) / num_words}.")
    print(f"Success rate 2 is {1 - len(failed2) / num_words}.")

#Success rate 1 is 0.9443416589577551.
#Success rate 2 is 0.9747918593894542.
#>>> len(failed)
#722
#>>> len(failed2)
#327
#>>> pickle.dump((failed, failed2), open("to_solve.pickle", "wb"))
failed1, failed2 = pickle.load(open("to_solve.pickle", "rb"))
fs1 = set([w for w, _ in failed2])
failed3 = retrieve_false_results(sol_dic)
fs2 = set([w[-1] for _, (w,n) in failed3])

def wordle_assist(guess, hint, remaining_words, move2="", num=3):
    possible_words = filter_word_by_hints(guess, hint, remaining_words)
    next_move = []
    move2 = [move2] if move2 else []
    if num == 0:
        print(f"I recommend 'stare'.")
        return "stare"
    if num == 1:
        for w1 in words:
            for w2 in good_words:
                #next_move.append((calculate_pair_possibilities3_with_dscore(w1, w2, possible_words), w1, w2))
                next_move.append((score_pair_distribution(w1, w2, possible_words), w1, w2))
        _, move, move2 = sorted(next_move)[0]
        print(f"I recommend '{move}' and perhaps next '{move2}'.")
        return possible_words, move
    for w in (possible_words + move2):
        #next_move.append((calculate_word_possibilities3_with_dscore(w, possible_words), w))
        next_move.append((score_hint_distribution(w, possible_words), w))
    _, move = sorted(next_move)[0]
    print(f"I recommend '{move}'.")
    return possible_words, move

if False:
    broken = []
    winners = []
    stats = []
    killers = Counter() #set() # {'cigar', 'sissy', 'rebut'}
    total = len(words)
    possible_solutions = [solution for wordset in best_quintuples_unique_dict.values() for solution in wordset]
    for solution in possible_solutions:
        solved = 0
        failed = 0
        options_left = 0
        wtf = [] ## It's empty.
        success = True
        for goal in words:
            #hints = all_hints(solution, goal)
            #green_hints = hints[0]
            #yellow_hints = hints[1]
            #green_filtered_words = filter_greens(green_hints, sgb_words)
            #yellow_filtered_words = filter_yellows(yellow_hints, green_filtered_words)
            possible_words = filter_solution_by_goal(solution, goal, words)
            num_possible_words = len(possible_words)
            if num_possible_words == 1:
                if possible_words[0] == goal:
                    #solved.append(goal)
                    solved = solved + 1
                    options_left = options_left +  1
                    #print(f"{goal} is solved")
                else:
                    wtf.append(goal)
                    print(f"{goal} is not {yellow_filtered_words[0]}!")
            else:
                #position_filtered_words = filter_yellow_positions(yellow_hints, yellow_filtered_words)
                #num_options = len(position_filtered_words)
                #if num_options == 1:
                #    #solved.append(goal)
                #    solved = solved + 1
                #    options_left = options_left +  1
                #else:
                killers[goal] = killers[goal] + 1
                failed = failed + 1
                options_left = options_left + num_possible_words
                if success:
                    broken.append((solution, goal, possible_words))
                    print(f"{goal} breaks {solution} with {num_possible_words} possibilities.")#: {position_filtered_words}.")
                success = False
                    #break
        if success:
            winners.append(solution)
            print(f"{solution} is victorious!")
        winrate = solved / total
        options_left = options_left / total
        print(f"{solution} has winrate {winrate} with {solved} solved and {failed} failures with on average {options_left} remaining possible words.")
        stats.append((solved, failed, winrate, options_left, solution))


#with open("tmps1.pickle", 'wb') as f:
#    pickle.dump((solved, mada), f)

# ///// #

#worddic = dict()
#worddic = dict((word, c) for c, word in enumerate(words))
#hintdic = dict()
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

def match_gills(w, l): # So guessing 't' in any but the first position is useless as it'll be yellow in all *ests unless there are two ts
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


# Can I find all possible solutions for "vests" within *ests? -- I'll start with all solutions where 'vests' can be the 5th move!
# Or maybe all solutions for *ests within 4 moves, i.e., after the 5th move, every word is isolated.
#['bests', 'fests', 'gests', 'hests', 'jests', 'kests', 'lests', 'nests', 'pests', 'rests', 'tests', 'vests', 'wests', 'yests', 'zests']
#{'y', 'j', 'l', 'k', 'g', 'f', 'z', 't', 'r', 'b', 'n', 'h', 'v', 'p', 'w'}
hest = [w for w in words if w[1:] == 'ests'] # len = 15 
l0 = set([w[0] for w in hest]) # len = 15
shest = set(hest)
hills = [w for w in words if w[1:] == 'ills']
shills = set(hills) 
r0 = set([w[0] for w in hills]) # len = 19
# What if I try it for this? 
rw = gills(r0)
# Now there are 9 with 5, 952 with 4, 6202 with 3...
# Is it actually worse than before?
# Combinatorially, how can we get 18/19 letters in 5 moves?  We need at least 3.6 per move.

if False:
    hashes = set()
    pairs = []
    for w1 in rw[3]:
        for  w2 in rw[3]:
            hsh = "".join(sorted(match_gills(w1, r0) | match_gills(w2, r0)))
            if not hsh in hashes:
                hashes.update([hsh])
                pairs.append((len(hsh), w1, w2))

#>>> Counter(c for c, _, _ in spairs)
#Counter({7: 26170, 6: 24151, 5: 10642, 8: 8816, 4: 3264, 3: 689, 9: 277})
# If it's 3, then we'd have to score 15/19 in 3 moves?
# >>> spairs = sorted(pairs)
#spairs2 = [(c, w1, w2, get_hash(w1, w2)) for c, w1, w2 in spairs]
#>>> pickle.dump((hashes, spairs2), open("gills_pairs.pickle", 'wb'))
if False:
    hashes, spairs = pickle.load(open("gills_pairs.pickle", 'rb'))
    hashes3 = set()
    triples = []
    i = 0
    for _, w1, w2, hsha in spairs:
        if i % 1000 == 0:
            print(f"Step {i/1000} / 74.  We have {len(triples)} triples so far.")
        hshb = set(hsha)
        for w3 in rw[3]:
            hsh = "".join(sorted((match_gills(w3, r0) | hshb)))
            if not hsh in hashes and not hsh in hashes3:
                hashes3.update([hsh])
                triples.append((len(hsh), hsh, w1, w2, w3))
        i += 1
#striples = sorted(triples, reverse=True)
#pickle.dump((hashes3, striples), open("gills_triples.pickle", 'wb'))
if False:
    hashes, _ = pickle.load(open("gills_pairs.pickle", 'rb'))
    hashes3, triples = pickle.load(open("gills_triples.pickle", 'rb'))
    old_hashes = hashes | hashes3
    hashes4 = set()
    quads = []
    i = 0
    for _, hsha, w1, w2, w3 in triples:
        if i % 1000 == 0:
            print(f"Step {i/1000} / 315.  We have {len(quads)} quadruples so far.")
        hshb = set(hsha)
        for w4 in rw[3]:
            hsh = "".join(sorted((match_gills(w4, r0) | hshb)))
            if not hsh in old_hashes and not hsh in hashes4:
                hashes4.update([hsh])
                quads.append((len(hsh), hsh, w1, w2, w3, w4))
        i += 1
#I need to test for the presence of 15 letters, well, 14, t possibly twice.
#If I cross of 3 letters a move, then after 4 moves, I'll only have hit 12 letters, which is insufficient. 3*3+4 = 13 is also insufficient.
# 3*2 + 4*2 = 14 might work. 5 + 3*3 = 14
# There is only one word that hit 5 of these: ['glyph'] -- 'grypt' is disqualified due tot he final t.
# There are 26 that hit 4 of them.  There are 13 ways to get 9 letters in two words.
# There are 2664 words with 3 of these.
#>mw = gests(l0)
# So for the first move I need either:
# A) mw[5] ~ 'glyph'
# B) mw[4] x 2 ~ two+ of the words that hit 4
# Otherwise, there's no way I can cover them in 4 moves.  And as order doesn't matter, it's without loss of generality.

# The 3000 possibilities to cover 8 letters with two words.
#>mp4 = list(set([tuple(sorted([w1, w2])) for c, w1, w2 in [(len(match_ests(w1, l0) | match_ests(w2,l0)), w1, w2) for w1 in mw[4] for w2 in mw[4]] if c == 8 and w1 not in mw[5] and w2 not in mw[5]]))
# Let's solve the case starting with mw[5] first.  Move 1 is glyph or grypt.
#>m1 = {'glyph':get_hint_distribution_with_words('glyph', hest)} # Move 1
#glyph:{'bbbbb': ['bests', 'fests', 'jests', 'kests', 'nests', 'rests', 'tests', 'vests', 'wests', 'zests'], 'gbbbb': ['gests'], 'bbbby': ['hests'], 'bybbb': ['lests'], 'bbbyb': ['pests'], 'bbybb': ['yests']}
#>h1 = m1['glyph']['bbbbb'] # The 10 / 15 words remaining after move 1.  Now there are 3 moves to hit 9+ letters.
#>l1 = set([w[0] for w in h1]) # {'n', 't', 'j', 'z', 'v', 'w', 'k', 'b', 'f', 'r'} of len 10
#>mw1 = gests(l1) # mw[5] == set(), there are 13 that hit 4 and 478 that hit 3.
#>m1, h1, l1, mw1 = pickle.load(open("roundA.pickle", 'rb'))
#hit7 = list(set([tuple(sorted([w1, w2])) for c, w1, w2 in [(len(match_ests(w1, l1) | match_ests(w2,l1)), w1, w2) for w1 in mw1[4] for w2 in mw1[4]] if c == 7]))
# There are no pairs of mw[4] that hit 8 letters.  There are actually none that hit 7, either! 
# There are 37 that hit 6.  To get to 9, we need a 3+ letter word, then.  Which means, I can actually just look at ways to hit 9 from mw1[3]!
#>hit9 = list(set([tuple(sorted([w1, w2, w3])) for c, w1, w2, w3 in [(len(match_ests(w1,l1) | match_ests(w2,l1) | match_ests(w3,l1)), w1, w2, w3) for w1 in mw1[3] for w2 in mw1[3] for w3 in mw1[3]] if c == 9]))
#>hit9 = pickle.load(open('hit9.pickle', 'rb')) # 2246 options.
# Let's make sure these are solutions!
#>>> s = []
#>>> f = []
#>>> for w2, w3, w4 in hit9:
#    ...  bb = max([c for c, _ in get_history_buckets(play_history(['glyph', w2, w3, w4], hest))])
#    ...  if bb == 1:
#        ...   s.append((w2,w3,w4))
#        ...  else:
#            ...   f.append((bb, (w2,w3,w4)))
# Yup, f is empty.  
# So I now have all solutions of type (A) solving *ests on the 5th move.
#>hist9a = [('glyph', w2, w3, w4) for w2, w3, w4 in hit9]
# The next question is, how's this turn out for ALL words?  
# If I can solve *ests on the 5th move.  And there's no way to solve *ests on the 4th move.  That would require 2x 5 letter hitting words, but there is only one.
# To be able to solve *ests by the 6th move, in all of these cases, I'll need the *ests to be in buckets that can be solved with only one more move for #5.
if False:
    #candidates4, candidates5 = pickle.load(open("tmp_cands.pickle", 'rb'))
    unsolvable = []
    solvable = []
    for history in hist9a:
        hist_buckets = get_history_buckets(play_history(history, words))
        hard_buckets = sorted((wl for c, wl in hist_buckets if shest & set(wl) and len(wl) > 2), reverse = True)
        solved = True
        for wl in hard_buckets:
            if sorted([(score_hint_distribution(w, wl), w) for w in words])[0][0] > 1:
                unsolvable.append( (history, wl) )
                solved = False
                print(f"* * * {history} is not solvable")
                break
        if solved:
            print(f"{history} is solvable.")
            solvable.append(history)
# OH!  I messed up.  There are solvable ones.  Now what?  Well, see how the solution works for every other bucket!
#  ('glyph', 'bawks', 'terfe', 'zanja') is broken by (2, ['torus', 'tiros', 'tirrs', 'torcs', 'toros', 'torrs', 'torts', 'turds', 'turms', 'turrs']) and (3, ['video', 'modem', 'cided', 'cimex', 'codec', 'coded', 'codex', 'cooed', 'coved', 'coxed', 'cumec', 'diced', 'dived', 'domed', 'doved', 'doxed', 'duded', 'equid', 'ivied', 'mimed', 'mimeo', 'mixed', 'mooed', 'moved', 'muxed', 'odeum', 'ummed', 'viced'])
# ('glyph', 'bawks', 'terfe', 'zanja') is broken by (2, ['tiars', 'trads', 'trams', 'trass', 'trats', 'tsars'])
if False: # Repeat of the above over all words :-3
    solvable_old, _ = pickle.load(open("hist9a_proof.pickle", 'rb'))
    unsolvable = []
    solvable = []
    for history in solvable_old:
        hist_buckets = get_history_buckets(play_history(history, words))
        hard_buckets = sorted((wl for c, wl in hist_buckets if len(wl) > 2), reverse = True)
        solved = True
        for wl in hard_buckets:
            if sorted([(score_hint_distribution(w, wl), w) for w in words])[0][0] > 1:
                unsolvable.append( (history, wl) )
                solved = False
                print(f"* * * {history} is not solvable")
                break
        if solved:
            print(f"{history} is solvable.")
            solvable.append(history)

# Wow.  None are solvable.  All fail on pretty much the same bucket: 
#>>> unsolvable[1969]
#(('glyph', 'bevor', 'funks', 'towzy'), ['temes', 'tests', 'tetes', 'texes', 'texts', 'tices', 'tides', 'times', 'tomes', 'toses', 'totes'])
# This was every possibility starting with the 5-letter word, "glyph"
#pickle.dump(unsolvable, open("hist9a_proof.pickle", "wb"))
#pickle.dump((solvable, unsolvable), open("hist9a_proof.pickle", "wb"))
#pickle.dump(unsolvable, open("hist9a_proof_hard.pickle", "wb"))
# -- Okay, so those 4 letter combos starting with 'glyph' that solve *ests don't solve the whole thing.

# B) mw[4] x 2 ~ two+ of the words that hit 4
# Okay, now on to (B).  There's no way to start with glyph and cover the 14 letters needed with a solution.
# Hmm, it's not a perfect proof.  Blegh.  But it's pretty close.  It says that We can devote 4 moves to solving *ests.  
# But, yeah, maybe 5 moves could solve *ests and everything else even though it doesn't in 4.
# So, yeah, I really do want to be looking at every possible way to solve *ests.
# Candidates can be any size, but will be at least length 4.  The problem is they could in principle be of length 5 and 'work' everywhere.
# However, I think I can only search through words that fill at least one character from l0-hest.  Otherwise, just have the length 4 solution.
#>hwords = list(mw[1]) # At least one letter
# At each step, I would want to only enumerate through words that hit at least one new one, too.
# I have 5 words to get 14, so if there are more than 5 remaining by word 4, I'm fucked.  If there are 4 remaining, it's only 'glyph', lol.  Which, actually, will be there for word #1, so I can skip it, too.
# Thus by word 4, I need to have at most 4 letters left.  If the cut_off is 5 then I need at least 4.
# And if there are 0 or 1 letters left with word 4, then I win :D
# What about by word number 3?  I have two words left to get 9+ letters.  Some possibilities exist.   
# If there are 8 letters left to get (meaning 9 total), assuming I can get at most 4 in the next round, I need to get at least 4 in this round.
# So the formula would be {remaining - 4}
# What about by the second word? ... Well, assuming 'glyph' is w1 or w2, we can score at most 12... so if it's more than that, yeah, fuck it. 
# ... okay, not even one word in overnight.  I need a smarter way.  Look for the confounding factors in the hist9a case?
# Any solution must also work there, which can further reduce the space of possibilities?
if False:
    candidates4 = []
    candidates5 = []
    for i1, w1 in enumerate(hwords): #enumerate(words):
        best_score = 0
        m1 = match_ests(w1, l0)
        l1 = l0 - m1
        mw1 = [w for w in hwords[i1+1:] if len(match_ests(w, l1)) > 0]
        for i2, w2 in enumerate(mw1):#enumerate(words[i1+1:]):
            m2 = match_ests(w2, l1)
            l2 = l1 - m2
            if len(l2) > 12: # Say m1 matches 1 and m2 also matches 1, then l3 is of size 15 - 2 == 13.  I have 3 more words, so I'm fucked
                continue
            else:
                mw2 = [w for w in mw1[i2+1:] if len(match_ests(w, l2)) > 0]
                for i3, w3 in enumerate(mw2): #enumerate(words[i2+1:]):
                    m3 = match_ests(w3, l2)
                    l3 = l2 - m3
                    remaining = len(l3) - 1
                    if remaining > 8:
                        continue
                    else:
                        cut_off3 = max(remaining - 4, 0)
                        mw3 = [w for w in mw2[i3+1:] if len(match_ests(w, l3)) > cut_off3]
                        for i4, w4 in enumerate(mw3): #enumerate(words[i3+1:]):
                            m4 = match_ests(w4, l3)
                            l4 = l3 - m4
                            cut_off4 = len(l4) - 1
                            if cut_off4 > 4:
                                continue
                            elif cut_off4 >= 1:
                                mw4 = [w for w in mw3[i3+1:] if len(match_ests(w, l4)) >= cut_off4]
                                for i5, w5 in enumerate(mw4): #enumerate(words[i4+1:]):
                                    m5 = match_ests(w5, l0)
                                    score = len(m1 | m2 | m3 | m4 | m5) 
                                    if score > best_score:
                                        best_score = score
                                    if score >= 14:
                                        candidates5.append((score,(w1, w2, w3, w4, w5)))
                            else: 
                                score = len(m1 | m2 | m3 | m4) 
                                if score > best_score:
                                    best_score = score
                                candidates4.append((score,(w1, w2, w3, w4)))
        print(f"Best score for {w1} ({i1}) is {score}.") 


def get_hc(word, possible_words):
    hints = dict()
    counts = Counter()
    for goal in possible_words:
        hint = get_hints3(word, goal)
        hints[hint] = hints.get(hint, []) + [goal]
        counts[hint] += 1
    return hints, counts

# Hard mode guesses.  Only constraints are that greens must match and yellows must be present
# I use it like this, however... So 
#possible_guesses = get_hc(guess, gwords)[0][hint]
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

# Okay, it's been long enough.
# Let's just brute force it.  
# How do I really just brute force the NP-complete problem?
# Well, I'd want to go through each starting word.
# Which is word_distros, right?
# Then for each hint in that, I want to try every single second word... and take the best, right?
# So, really, there's no such thing as a guess.
# bwords are the words to split
#guess = 'cigar'; bwords = words[:100]
#>>> Counter(c for _, c in retrieve_all_results(sol_dico))
#Counter({4: 1557, 5: 430, 3: 283, 2: 45})
#>>> Counter(c for _, c in retrieve_all_results(sol_dicho))
#Counter({4: 1143, 3: 835, 5: 224, 2: 90, 6: 23})
#>>> si_sols3, fails3 = pickle.load(open("si_sols3.pickle", 'rb'))
#>>> Counter(c for _, c in retrieve_all_results(si_sols3))
#Counter({4: 6451, 5: 5113, 3: 763, 6: 614, 2: 21, 7: 7, 8: 3})
# Okay, I know that I can solve everything in at most 8 words.

# I realized I can store as tuples the wordsets that can't be solved.
# The lack of the words in moves 1-3 are okay because by construction they won't disambiguate this wordset.
#move4 = dict()
#move5 = dict()
#move4 = pickle.load(open("move4.pickle", 'rb'))
#move5 = pickle.load(open("move5.pickle", 'rb'))
move4_count = 0
move3_count = 0
#moves = dict((i, dict()) for i in range(1,6))
#moves = pickle.load(open("moves.pickle", 'rb'))
# What if I make smoves for solutions?
moves, smoves, unsolved, solved_indices = pickle.load(open("sdata2.pickle", 'rb'))
moves = dict((i, dict()) for i in range(1,6))
smoves = dict((i, dict()) for i in range(1,6))
move_counts = dict((i, 0) for i in range(1,6))
smove_counts = dict((i, 0) for i in range(1,6))

def dump4():
    pickle.dump(move4, open("move4.pickle", "wb"))
def dump5():
    pickle.dump(move5, open("move5.pickle", "wb"))
def dumpmoves():
    pickle.dump(moves, open("moves.pickle", "wb"))
def recupdate(moves1, moves2):
    for i, v in moves1.items():
        v.update(moves2[i])
def updatemoves(moves2):
    #moves2 = pickle.load(open("moves.pickle", 'rb'))
    recupdate(moves, moves2)
    #dumpmoves()

def brute_force_wordle(bwords, gwords, move=1, history=[], hard=False, limit=69, fixed_start=None, checkthree=False, order=False):
    #global move3_count
    #global move4_count
    best_score = len(bwords) + move
    solution = dict()
    good_move = None
    rmove = limit - move
    if rmove <= 0: # move >= limit:
        return best_score, (move, 'infinity', ",".join(bwords), history)
    # check if it's already been broken
    if rmove <= 2 or (checkthree and rmove <= 4): #move >= limit - 2:
        key = tuple(sorted(bwords)); #d = limit - move
        if key in moves[rmove]:
            move_counts[rmove] += 1
            return best_score, solution
        if key in smoves[rmove]:
            good_move = smoves[rmove][key]
            smove_counts[rmove] += 1
    if move < 5: # I don't like the text to scroll by too fast . . .
        print(f"Move {move} ({history+['*']}) trying to solve {len(bwords)} words, starting with {bwords[:5]}.")
    '''
    if move == limit - 1: # 5, generally
        key = tuple(sorted(bwords))
        if key in move5:
            move5_count += 1
            return best_score, solution
    elif move == limit - 2: # 4, generally
        key = tuple(sorted(bwords))
        if key in move4:
            move4_count += 1
            return best_score, solution
    '''
    lwords = len(bwords)
    if fixed_start:
        i = gwords.index(fixed_start)
        to_guess = [fixed_start]; gwords = to_guess + gwords[:i] + gwords[i+1:]
    else:
        if good_move in gwords:
            i = gwords.index(good_move)
            to_guess = [good_move] + gwords[:i] + gwords[i+1:]
        elif order and move >= 2:
            to_guess = gwords
            to_guess = [t[1] for t in sorted([(score_hint_distribution(word, bwords), word) for word in to_guess])]
            '''
            if 'polar' in to_guess:
                del to_guess[to_guess.index('polar')]; to_guess.append('polar')
            if 'poral' in to_guess:
                del to_guess[to_guess.index('poral')]; to_guess.append('poral')
            if 'loran' in to_guess:
                del to_guess[to_guess.index('loran')]; to_guess.append('loran')
            if 'parol' in to_guess:
                del to_guess[to_guess.index('parol')]; to_guess.append('parol')
            if 'morae' in to_guess:
                del to_guess[to_guess.index('morae')]; to_guess.append('morae')
            if 'realo' in to_guess:
                del to_guess[to_guess.index('realo')]; to_guess.append('realo')
            if 'roues' in to_guess:
                del to_guess[to_guess.index('roues')]; to_guess.append('roues')
            if 'rouen' in to_guess:
                del to_guess[to_guess.index('rouen')]; to_guess.append('rouen')
            if 'ureal' in to_guess:
                del to_guess[to_guess.index('ureal')]; to_guess.append('ureal')
            if 'porae' in to_guess:
                del to_guess[to_guess.index('porae')]; to_guess.append('porae')
            if 'pareu' in to_guess:
                del to_guess[to_guess.index('pareu')]; to_guess.append('pareu')
            if 'pareo' in to_guess:
                del to_guess[to_guess.index('pareo')]; to_guess.append('pareo')
            if 'oaked' in to_guess:
                del to_guess[to_guess.index('oaked')]; to_guess.append('oaked')
            if 'uraos' in to_guess:
                del to_guess[to_guess.index('uraos')]; to_guess.append('uraos')
            '''
            '''
            if move == 2:
                good_move = "swift"
                i = gwords.index(good_move)
                to_guess = [good_move] + gwords[:i] + gwords[i+1:]
            '''
        else:
            to_guess = gwords
        gwords = to_guess
    for i, guess in enumerate(to_guess):
        tmp_history = history + [guess]
        hint_dist, counts = get_hc(guess, bwords)
        score = counts.most_common(1)[0][1]
        #print(f"{guess} ~ {counts}")
        #print(hint_dist)
        if score == 1:
            #print(f"{guess} is done.")
            for hint, wordlist in hint_dist.items():
                word = wordlist[0]
                if guess == wordlist[0]:
                    solution[hint] = (move, word, tmp_history)
                    best_score = move
                else:
                    solution[hint] = (move+1, word, tmp_history+[word])
                    best_score = move+1
            if rmove <= 2 or (checkthree and rmove <= 4):
                smoves[rmove][key] = guess
            return best_score, solution
        if not fixed_start and (score == lwords or move == limit - 1):
            continue
        # So it's not solved.  We need to try every word on each hint!
        next_scores = dict()
        worst_score = move
        for hint, wordlist in hint_dist.items():
            lwd = len(wordlist)
            if lwd > 1:
                if hard:
                    possible_guesses = get_hg(guess, hint, gwords) #get_hc(guess, gwords)[0][hint]
                else:
                    possible_guesses = gwords[i+1:]
                score, next_score = brute_force_wordle(wordlist, possible_guesses, move=move+1, history=tmp_history, hard=hard, limit=limit, checkthree=checkthree, order=order)
                #if score > limit:
                #    print(f"{score}: {next_score}")
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
            if worst_score <= limit: # Should probably read: worst_score <= limit.  If I'm looking for ANY solution within the limit.  Basically, I found a good enough word.  No need to find the best!
                # It's ugly, but this means we've found a solution, right?
                if rmove <= 2 or (checkthree and rmove <= 4):
                    smoves[rmove][key] = guess
                return best_score, solution
    if rmove <= 2 and best_score > limit:
        if rmove == 2:
            #move4_count += 1
            print("%s ~ ~%s" % (best_score, solution))
        moves[rmove][key] = True
    elif best_score > limit:
        #move3_count += 1
        print("%s, %s ~ ~%s" % (move, best_score, solution))
        if checkthree and rmove <= 4:
            moves[rmove][key] = True
    '''
    if not solution: 
        if move == limit - 1:
            move5[key] = True
    elif move == limit - 2 and best_score > limit:
            move4_count += 1
            move4[key] = True
            print("%s ~ ~%s" % (best_score, solution))
        #return best_score, (move, 'infinity', ",".join(bwords), history)
    '''
    return best_score, solution

zwords = pickle.load(open("zwords.pickle", "rb")) # Some random and some of the better words first... for, yea, better guessing.
#zwords = pickle.load(open("zwords2.pickle", "rb")) 
#score, best = brute_force_wordle(official_goals, zwords, move=1, limit=4) # I actually know that 5 is enough!
#pickle.dump((score, best), open("final_solution_in_4.pickle", 'wb'))
#score, best = pickle.load(open("final_solution_in_5.pickle", 'rb'))
print()
print()
print("-- -- --")
print("Solving all words.")
print("-- -- --")
print()
print()
#scorea, besta = brute_force_wordle(words, zwords, move=1, limit=6) # I know I can solve it within 8 moves.
#pickle.dump((scorea, besta), open("final_solution_all_in_6.pickle", 'wb'))

hw3 = ['zills', 'maxes', 'gages', 'pents', 'lamer', 'jefes', 'memes', 'fents', 'namer', 'bases', 'zests', 'mages', 'peeps', 'vives', 'waxes', 'yeses', 'pixes', 'waker', 'loges', 'jells', 'zexes', 'games', 'faxes', 'waxer', 'waffs', 'waler', 'tills', 'bills', 'vises', 'wizes', 'maker', 'tates', 'faffs', 'mills', 'vells', 'vills', 'bayes', 'eases', 'rares', 'sages', 'wills', 'fames', 'zerks', 'dills', 'gills', 'hajes', 'serks', 'babes', 'tawer', 'river', 'faves', 'fanes', 'pills', 'jakes', 'lills', 'yexes', 'mazer', 'kexes', 'nills', 'gazes', 'sates', 'pizes', 'pipes', 'lawer', 'mazes', 'loves', 'wawes', 'fills', 'sazes', 'kills', 'fezes', 'tests', 'seeps', 'sills', 'sakes', 'fests', 'zezes', 'wises', 'zaxes', 'jaker', 'hazes', 'fazes', 'leges', 'yills', 'jives', 'tents', 'fazed', 'mells', 'waqfs', 'pests', 'hills', 'fakes', 'vaxes', 'janes', 'faked', 'taver', 'wiver', 'viver', 'raker', 'rills', 'wakfs', 'laxer', 'leves', 'cills', 'gares', 'wafer', 'jills', 'james']
#score, best = brute_force_wordle(hills, zwords, move=1, hard=True, limit=6) # I actually know that 5 is enough!
#score, best = brute_force_wordle(hw3, zwords, move=1, hard=True, limit=6) # I actually know that 5 is enough!

#_,_,_, ww = pickle.load(open("hw3_full_sols_words.pickle", 'rb'))
#ww, diffs = pickle.load(open("word_difficulties.pickle", "rb"))

ww, diffs = pickle.load(open("word_difficulties3.pickle", "rb")) 
rww, _,  _ = pickle.load(open("rww2_and_ww.pickle", "rb")) # rww2, rww, ww

#start = time.time()
#score, best = brute_force_wordle(list(set(ww[2] + ww[1] + ww[19])), zwords, move=1, limit=6, fixed_start='salet', hard=True)
#end = time.time()
#print(f"It took {end - start} seconds.")

# What if for each word, I look for an unsatisfiable core?
#>>> ww[7] = [w for w in words if w[1:]== 'ills']                                                          

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

if False:
    diffs = dict()
    for word in words:
        diffs[word] = get_ww_diff(word, ww, 7)
    pickle.dump( (ww, diffs), open("word_difficulties3.pickle", 'wb'))

# atm, on DELL, the top is easy and the bottom is hard.
# I'll do the same from HP but in the opposite direction
# Okay.  I should just solve hard first as it's faster and less reduntant

def dumpunsolved(unsolved):
    pickle.dump(unsolved, open("unsolved.pickle", "wb"))
def updateunsolved(unsolved2):
    #unsolved2 = pickle.load(open("unsolved.pickle", 'rb'))
    for word, info in unsolved2.items():
        if not word in unsolved: 
            unsolved[word] = info
    #dumpunsolved(unsolved)
def updateindices(solved_indices2):
    #solved_indices2 = pickle.load(open("solved_indices.pickle", "rb"))
    for word, indices in solved_indices2.items():
        if word in solved_indices: 
            if len(solved_indices[word]) < len(indices):
                solved_indices[word] = indices
        else:
            solved_indices[word] = indices
    #pickle.dump(solved_indices, open("solved_indices.pickle", "wb"))
#def saveprogress(unsolved):
#    updatemoves()
#    updateindices()
#    updateunsolved(unsolved)

def dumpdata():
    pickle.dump((moves, smoves, unsolved, solved_indices), open("sdata2.pickle", 'wb'))

def saveprogress():
    moves2, smoves2, unsolved2, solved_indices2 = pickle.load(open("sdata2.pickle", 'rb'))
    updatemoves(moves2)
    updatemoves(smoves2)
    updateindices(solved_indices2)
    updateunsolved(unsolved2)
    dumpdata()

#>>> len(words) / 4
#3243.0
inc = int(len(words) / 4)
words1 = words[:inc]; del words1[words1.index('puppy')]; words1.insert(0, 'puppy'); del words1[words1.index('aider')]; words1.insert(1, 'aider'); del words1[words1.index('crane')]; words1.insert(0, 'crane')
words2 = words[inc:2*inc]; del words2[words2.index('fuffy')]; words2.insert(0, 'fuffy'); del words2[words2.index('deair')]; words2.insert(1, 'deair')
words3 = words[2*inc:3*inc]; del words3[words3.index('nunny')]; words3.insert(0, 'nunny'); del words3[words3.index('oared')]; words3.insert(0, 'oared')
words4 = words[3*inc:4*inc]; del words4[words4.index('yuppy')]; words4.insert(0, 'yuppy'); del words4[words4.index('redia')]; words4.insert(1, 'redia'); del words4[words4.index('salet')]; words4.insert(0, 'salet');
'''
del zwords[zwords.index('zanja')]; zwords.insert(5, 'zanja')
del zwords[zwords.index('zymic')]; zwords.insert(5, 'zymic')
del zwords[zwords.index('vying')]; zwords.insert(5, 'vying')
del zwords[zwords.index('swift')]; zwords.insert(0, 'swift')
del zwords[zwords.index('toner')]; zwords.insert(0, 'toner')
del zwords[zwords.index('ducal')]; zwords.insert(0, 'ducal')
del zwords[zwords.index('loper')]; zwords.insert(0, 'loper')
del zwords[zwords.index('moral')]; zwords.insert(0, 'moral')
del zwords[zwords.index('omber')]; zwords.insert(0, 'omber')
#del zwords[zwords.index('oared')]; zwords.insert(0, 'oared')
#del zwords[zwords.index('hoaed')]; zwords.insert(0, 'hoaed')
#del zwords[zwords.index('lemur')]; zwords.insert(0, 'lemur')
del zwords[zwords.index('salet')]; zwords.insert(13, 'salet')
'''
bsol = pickle.load(open("swift_bsol_words.pickle", 'rb'))
#sbwords2 = set(get_sol_entries(bsol['ybbbb']))
#score, best = brute_force_wordle(zwords, zwords, move=1, limit=8, fixed_start='swift', checkthree=False, order=True)
from random import shuffle
if False:
    #words.reverse()
    #unsolved = pickle.load(open("unsolved.pickle", "rb")) #dict()
    solved = dict()
    #solved_indices = pickle.load(open("solved_indices.pickle", "rb"))
    ww.update(rww) # zymic is getting so close I wanted to set it up to cruise on to a full solution!
    lww = len(ww)
    #words4.reverse() 
    #for word in words4: # later to be words, lol
    zwords = [t[1] for t in sorted([(score_hint_distribution(word, ww[7]), word) for word in words])]
    for word in zwords[:10]:#, 'glyph', 'gyppy']: # gyppy is one of the worst words!
    #for word in ['vifda', 'zanja', 'skint']: # Some of the best words!
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
        solved_indices['swift'] = []
        #solved_indices['swift'] = [33, 51, 46, 38, 39, 40, 52, 53, 36, 55, 34, 35, 37, 41, 42, 43]
        #wdiff.reverse()
        #zwords.reverse()
        #for i in [7, 27]: #diffs[word]: # Now it will continue to step through the indices for every single word's set!
        #lrww = list(rww.keys())[2:]
        #for i in [i for score_avg, score_count, i in wdiff] + lrww: #list(rww.keys()): #diffs[word]: # Now it will continue to step through the indices for every single word's set!
        #inds = [i for score_avg, score_count, i in wdiff] + list(rww.keys()) #diffs[word]: # Now it will continue to step through the indices for every single word's set!
        #ainds = [31, 32]
        #binds = [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]
        #binds = [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 56]
        #binds = [34, 36, 33, 35, 56, 51, 57, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55]
        #binds =  [28, 33, 51, 46, 38, 39, 40, 52, 53, 36, 55, 34, 35, 37, 41, 42, 43, 44, 45, 47, 48, 49, 50, 54, 56, 57]
        binds = [7]
        #inds.reverse()
        #shuffle(binds)
        for i in binds:
        #for i in [28]:
            num += 1
            print(f"++ Starting index {num}/{lww} of {word}. ({indices})")
            ws = set(ww[i])
            indices.append(i); 
            bwords.update(ww[i])
            if i in solved_indices[word]:
                continue
            #if ws.issubset(bwords):
            #    continue
            count = 0
            if tryHeuristic:
                best = compute_all_wordles(word, bwords, move=1, limit=6)
                if check_results(best):
                    solved_indices[word].append(i)
                    print(f"- solved by greedy strategy.")
                    continue
                tryHeuristic = False
            sbwords = sorted(bwords); lbw = len(sbwords)
            score, best = brute_force_wordle(sbwords, zwords, move=1, limit=6, fixed_start=word, hard=hard, checkthree=(lbw < 300), order=True)
            #if hard and score > 6: # idea: if it can be solved in hard mode, it can be solved in easy mode.  Only try easy mode when hard is broken!
            #    print("--> easy") # ... and then in remaining loops, no point in bothering with hard!
            #    hard = False
            #    score, best = brute_force_wordle(list(bwords), zwords, move=1, limit=6, fixed_start=word)
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

# >>> retrieve_all_results(sol['ybbbb']['bbbgb']['bbbbg'])
#sol = pickle.load(open("swift_sol_words.pickle", 'rb'))

# omfg, the following worked!
#>>> get_fails(sol['ybbbb']['bbbgb'])[0]
#{'bbbbg': {'hexes', 'veges', 'zexes', 'zezes', 'kexes', 'vexes'}, 'bgbyg': {'doves', 'doxes'}, 'bgbbg': {'coxes', 'nones', 'coses', 'jones', 'coves', 'voces'}}
#sbwords = get_sol_entries(sol['ybbbb']['bbbgb']['bbbbg'])
#score, best = brute_force_wordle(sbwords, zwords, move=4, limit = 6, hard=False, checkthree=True)
#best2 = fixup_history(best, ['swift', 'pareu', 'molds'])
#>>> sbwords =  get_sol_entries(sol['ybbbb']['bbbgb']['bgbyg'])
#>>> sbwords
#['bodes', 'codes', 'nodes', 'dobes', 'doges', 'doses', 'doves', 'doxes', 'dozes']
#>>> 
#Oh, holy shit, this killed it!
#sbwords =  get_sol_entries(sol['ybbbb']['bbbgb']) # ... len 156

#sbwords = get_sol_entries(sol['ybbbb']['bgbgb']) # ... 
#>>> get_fails(sol['ybbbb']['bgbgb'])[0]
#{'bybbb': {'laxes', 'lazes'}, 'bbbyb': {'mazes', 'maxes'}}
#sbwords = get_sol_entries(sol['ybbbb']['bgbgb']['bybbb']) # failed! whoa
#['easel', 'dales', 'eales', 'gales', 'hales', 'kales', 'yales', 'lades', 'lased', 'lakes', 'lanes', 'lases', 'laxes', 'lazes', 'laves', 'vales']
#sbwords = get_sol_entries(sol['ybbbb']['bgbgb']['bbbyb']) # ALSO FAILED!
#['dames', 'hames', 'james', 'kames', 'games', 'mages', 'makes', 'mases', 'maxes', 'mazes', 'manes', 'mased', 'names']
#zwords = [t[1] for t in sorted([(score_hint_distribution(word, sbwords), word) for word in words])]
#score, best = brute_force_wordle(sbwords, zwords, move=3, limit = 7, hard=False, checkthree=True, history = ['swift', 'pareu'])

#score, best = brute_force_wordle(sbwords, zwords, move=3, limit = 6, hard=False, checkthree=True, history = ['swift', 'pareu'])
#sbwords = get_sol_entries(sol['bbbbb']['bbbby']['bgbyb']) # -- yay
#sbwords = get_sol_entries(sol['bbbbb']['bbbby']['bgbbb']) # -- yay
#sbwords = get_sol_entries(sol['bbbbb']['bbbyy']['bgbbg']) # len 12 ... NO
#sbwords = get_sol_entries(sol['bbbbb']['bbbyy']) # ... YES!
#score, best = brute_force_wordle(sbwords, zwords, move=3, limit = 6, hard=False, checkthree=True, history = ['swift', 'rayle'])

#sbwords = get_sol_entries(sol['ybybb']['ybbbb']) # len 30 ... YES!

#>>> get_fails(bsol['ybbby'])[0]
#{'bbbgy': {'kests', 'zests', 'vents', 'tests', 'tents'}, 'bbyyy': {'tajes', 'taxes'}}
#sbwords = get_sol_entries(sol['ybbby']['bbyyy']) # len 30, too . . . YES!
#sbwords = get_sol_entries(sol['ybbby']['bbbgy']) # len 49. 

# Hold on, it seems as if I'll solve all of ['ybbby'] faster... wtf
# Yeah, it worked.
#sbwords = get_sol_entries(sol['ybbby']) # len 562

#zwords = [t[1] for t in sorted([(score_hint_distribution(word, sbwords), word) for word in words])]
#score, best = brute_force_wordle(sbwords, zwords, move=1, limit=6, hard=False, checkthree=True, fixed_start='swift')
#score, best = brute_force_wordle(sbwords, zwords, move=3, limit=6, hard=False, checkthree=True, history = ['swift', 'orate'], order=True)
bsol = pickle.load(open("swift_bsol_words.pickle", 'rb'))
sbwords = get_sol_entries(bsol['ybbbb'])
print(len(sbwords))
#score, best = brute_force_wordle(sbwords, zwords, move=1, limit=7, hard=False, checkthree=True, fixed_start='swift', order=True)

#score, best = brute_force_wordle(zwords, zwords, move=1, limit=6, fixed_start='swift', checkthree=True, order=True)

# Now that I ordered everything from move 2+
# I might as well just try the greedy-first brute-force... meh.

## Okay, the above is actually seeming likely to solve everything for Zymic.
## Oddly, Zymic was broken by some subset during greedy but now is solving it with more... weird.  Well, it's non-deterministic.  The brute-force search is the deterministic one, in terms of the decision procedure result.
#>>> all_ww = list(set([w for wl in ww.values() for w in wl]))
#>>> remain_ww = dict((i, [c[0] for c in retrieve_all_results(si_sols3) if c[1] == i and not c[0] in all_ww]) for i in range(2,9))
#>>> 
#>>> Counter([c[1] for c in retrieve_all_results(si_sols3)])
#Counter({4: 6451, 5: 5113, 3: 763, 6: 614, 2: 21, 7: 7, 8: 3})
#>>> [len(remain_ww[i]) for i in range(2,9)]
#[20, 687, 6095, 4745, 378, 2, 0]
# Okay, if by chance Zymic beats the hard set, for now, I'll just go in reverse order over everything in the Salet solution that's not in ww.
#pickle.dump((rww, ww), open("rww_and_ww.pickle", "wb"))
#pickle.dump((rww2, rww, ww), open("rww2_and_ww.pickle", "wb"))
'''
>>> rww[31] = remain_ww[7] + remain_ww[6][:150]
>>> rww[32] = remain_ww[6][150:]
>>> for in range(33, 33+13):
>>>     rww[i] = remain_ww[5][(i - 33) * 365: (i - 33 + 1) * 365]
>>> for i in range(46, 46+23):
>>>     rww[i] = remain_ww[4][(i - 46) * 265: (i - 46 + 1) * 265]
>>> for i in range(69, 69+3):
>>>     rww[i] = remain_ww[3][(i - 69) * 229: (i - 69 + 1) * 229]
>>> rww[72] = remain_ww[2]
>>> len(all_rww)
>>> 11927
>>> wordset == set(all_rww + all_ww)
>>> True
>>> set(all_rww) & set(all_ww)
>>> set()

# I took the unsolved entries from a global Swift solution and brooke them up into etnries prior to rww... cuz Swift is almost done with the basic wws
>>> len(get_sol_entries(bsol['ybbbb']))                                                                                                                                                                            
2133                                                                                                                                                                                                               
>>> len(get_sol_entries(bsol['ybbby']))                                                                                                                                                                            
562    

# The smol one
>>> rww2[31] = a[:256]
>>> rww2[32] = a[256:] 

# The large one
dict_keys([31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59])
# 27 * 79 size bins

# So if it gets to 59, I'm done ;D
>>> for i, wl in rww.items():
    ...  rww2[i+30] = rww[i]
    ... 

'''

if False:
    solvers = []

    for word in ['chynd'] + list((wordset - set(['chynd']))):
        print(f"Trying {word}.")
        #sol = compute_all_wordles(word, hills)
        sol = compute_all_wordles_hard(word, hills, words, lookahead=-1)
        if check_results(sol):
            print(f"{word} solves *ILLS with {', '.join(sorted(f'{n} : {c}' for n, c in Counter([c[1] for c in retrieve_all_results(sol_dicho)]).items()))}.")
            solvers.append((word, sol))

#sol = pickle.load(open("chynd_sol_words.pickle", 'rb')); f, ff = get_fails(sol); hw = list(set(hills) | f['bbbbb'])
#sol2 = compute_all_wordles('chynd', hw, limit=9)
#hw2 = [r[0][-1] for r in retrieve_all_results(sol2) if r[1] >= 5 and r[0][1] == 'aleft']; hw2 = list(set(hw2 + hills))
#pickle.dump(sol0, open("brock_sol_words.pickle", 'wb'))
#{'bbbbb': {'wawes', 'zills', 'vexes', 'sazes', 'saves', 'mimes', 'eaves', 'memes', 'waxes', 'zests', 'peeps', 'vives', 'seeps', 'zexes', 'faxes', 'meves', 'zaxes', 'jests', 'fazes', 'wizes', 'jives', 'vells', 'vills', 'mells', 'wills', 'faves'}, 'bybbb': {'viver', 'rares', 'waxer', 'laxer', 'waler', 'lamer', 'mazer', 'tawer', 'namer', 'taver', 'river', 'lawer', 'wafer', 'wiver', 'gares'}, 'bbybb': {'golly', 'lolly'}, 'bbbyy': {'kacks', 'jacks'}}
#pickle.dump(sol0, open("lovie_sol_words.pickle", 'wb'))
#{'bbbyb': {'sings', 'jinks', 'wings', 'binks'}, 'bbbby': {'waker', 'zexes', 'games', 'zerks', 'janes', 'fazed', 'tents', 'pests', 'sages', 'zaxes', 'fezes', 'bases', 'tests', 'jakes', 'sates', 'eases', 'fazes', 'fames', 'pents', 'faxes', 'faked', 'fakes', 'serks', 'sakes', 'zests', 'mages', 'fents', 'fanes', 'zezes', 'bayes', 'fests', 'seeps', 'jefes', 'yexes', 'maxes', 'maker', 'peeps', 'tates', 'james', 'babes', 'yeses', 'mazes', 'hajes', 'hazes', 'sazes', 'wawes', 'waxes'}, 'bbbbb': {'jumps', 'sacks', 'jacks', 'zacks', 'mumps'}, 'bbbyy': {'cines', 'zines'}, 'ybbby': {'wells', 'jells'}, 'bgbby': {'cowed', 'goxes', 'oozes', 'ooses', 'yowes', 'cozed'}, 'bgbbb': {'jooks', 'gooks'}, 'ybbyb': {'sills', 'fills', 'zills', 'jills', 'pills'}}
#sol_vifda = compute_all_wordles('vifda', words, limit=9); counter = 0; sol_jived = compute_all_wordles('jived', words, limit=9); counter = 0; sol_lever = compute_all_wordles('lever', words, limit=9); counter = 0; sol_brink = compute_all_wordles('brink', words, limit=9); counter = 0; sol_skint = compute_all_wordles('skint', words, limit=9)
#pickle.dump((sols, fails, hw5, ww), open("hw3_full_sols_words.pickle", 'wb')) # Actually 4, I fucked up.
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
## Chosen via various word_classes....  Maybe over-kill, right?
## But I got sick of NOT failing my basic solvers.
#    hw2 = ['gages', 'waqfs', 'tills', 'jives', 'faffs', 'fills', 'leges', 'vises', 'vaxes', 'kills', 'vives', 'waffs', 'pills', 'dills', 'zaxes', 'pipes', 'tests', 'wises', 'leves', 'wizes', 'pixes', 'loves', 'jaker', 'zests', 'pizes', 'hills', 'kexes', 'zills', 'nills', 'wakfs', 'yills', 'loges', 'jills', 'memes', 'cills', 'mills', 'gazes', 'lills', 'sills', 'gills', 'rills', 'babes', 'raker', 'wills', 'bills', 'vells', 'vills', 'jells']
    # Hadda make a new one based on brock and lovie
#    hw3 = ['zills', 'maxes', 'gages', 'pents', 'lamer', 'jefes', 'memes', 'fents', 'namer', 'bases', 'zests', 'mages', 'peeps', 'vives', 'waxes', 'yeses', 'pixes', 'waker', 'loges', 'jells', 'zexes', 'games', 'faxes', 'waxer', 'waffs', 'waler', 'tills', 'bills', 'vises', 'wizes', 'maker', 'tates', 'faffs', 'mills', 'vells', 'vills', 'bayes', 'eases', 'rares', 'sages', 'wills', 'fames', 'zerks', 'dills', 'gills', 'hajes', 'serks', 'babes', 'tawer', 'river', 'faves', 'fanes', 'pills', 'jakes', 'lills', 'yexes', 'mazer', 'kexes', 'nills', 'gazes', 'sates', 'pizes', 'pipes', 'lawer', 'mazes', 'loves', 'wawes', 'fills', 'sazes', 'kills', 'fezes', 'tests', 'seeps', 'sills', 'sakes', 'fests', 'zezes', 'wises', 'zaxes', 'jaker', 'hazes', 'fazes', 'leges', 'yills', 'jives', 'tents', 'fazed', 'mells', 'waqfs', 'pests', 'hills', 'fakes', 'vaxes', 'janes', 'faked', 'taver', 'wiver', 'viver', 'raker', 'rills', 'wakfs', 'laxer', 'leves', 'cills', 'gares', 'wafer', 'jills', 'james']
#    hw4 = ['mazes', 'yards', 'yeses', 'hazer', 'wakfs', 'mills', 'vells', 'pipes', 'kawas', 'goxes', 'loves', 'fakes', 'gawks', 'balas', 'sooks', 'sates', 'taver', 'fates', 'poods', 'fuffs', 'fanes', 'lalls', 'mazer', 'poofs', 'haves', 'zooks', 'james', 'gamer', 'zaxes', 'raker', 'sacks', 'waxer', 'zexes', 'dazed', 'daled', 'jaded', 'burgs', 'waxes', 'gaffs', 'fines', 'rezes', 'jacks', 'kexes', 'wents', 'coyed', 'konks', 'lamer', 'vests', 'japes', 'fulls', 'songs', 'sails', 'kills', 'fezes', 'soops', 'fames', 'fazed', 'faxed', 'gives', 'verry', 'tills', 'sools', 'burrs', 'dawks', 'pizes', 'razer', 'jefes', 'kooks', 'mazed', 'herry', 'fests', 'keeks', 'faces', 'raxes', 'galls', 'wafer', 'foxes', 'faves', 'maxed', 'cools', 'waker', 'hakes', 'pangs', 'wawas', 'zills', 'noons', 'rills', 'tawer', 'vills', 'taxes', 'zoons', 'tests', 'rarks', 'wacks', 'puffs', 'oozes', 'jager', 'jests', 'teats', 'fents', 'famed', 'hills', 'jails', 'bayes', 'warps', 'gests', 'rexes', 'rager', 'hahas', 'pests', 'gazes', 'cills', 'viver', 'fazes', 'sowls', 'wakas', 'wolly', 'kacks', 'boded', 'wawes', 'babes', 'rares', 'ealed', 'lolly', 'gowls', 'maker', 'wases', 'fades', 'river', 'laver', 'fader', 'gores', 'games', 'wonks', 'leges', 'coxed', 'eases', 'lowes', 'boxed', 'plaps', 'zezes', 'jills', 'loges', 'geeps', 'bases', 'pixes', 'lills', 'faked', 'tents', 'gazer', 'leves', 'lulls', 'nines', 'galas', 'yills', 'jaggs', 'wizes', 'sinks', 'vives', 'mases', 'fares', 'sakes', 'mells', 'zacks', 'bakes', 'gares', 'zerks', 'jakes', 'fills', 'dazes', 'jades', 'doxes', 'vises', 'hajes', 'bills', 'seeps', 'zines', 'walls', 'sills', 'gades', 'sages', 'fangs', 'dills', 'barks', 'worts', 'yaffs', 'janes', 'ooses', 'gaged', 'sorts', 'sagas', 'pases', 'nills', 'souks', 'jouks', 'pages', 'fards', 'jells', 'namer', 'dazer', 'balls', 'yages', 'toped', 'laxer', 'jaker', 'jafas', 'paper', 'papes', 'woops', 'jarks', 'wakes', 'mages', 'nongs', 'waffs', 'rajes', 'memes', 'waler', 'faffs', 'baked', 'yexes', 'seats', 'faxes', 'fight', 'hight', 'rores', 'gapes', 'waqfs', 'vases', 'gases', 'jolts', 'jives', 'tajes', 'maxes', 'sazes', 'cozed', 'hazes', 'vaxes', 'gecks', 'wiver', 'eaves', 'pills', 'vades', 'winks', 'tozed', 'gages', 'babas', 'gazed', 'gills', 'zests', 'jagas', 'serks', 'jooks', 'kecks', 'slaps', 'pents', 'dopes', 'weeks', 'jarps', 'wills', 'wises', 'paxes', 'tolts', 'peeps', 'tates', 'lawer']
 
if False:
    word_distros = pickle.load(open("word_distros.pickle", "rb"))
    #hills_solvers = [t[0] for t in pickle.load(open("hills_solvers.pickle", 'rb'))]
    #hills_solvers = [t[0] for t in pickle.load(open("hills_solvers2.pickle", 'rb'))]
    hills_solvers = [t[0] for t in pickle.load(open("hills_solvers3.pickle", 'rb'))]
    _, _, hw5 = pickle.load(open("hw4_full_sols_words.pickle", 'rb'))
    solvers = []
    for word in hills_solvers[28:]:
        count = 0
        print(f"Trying {word}.")
        sol = compute_all_wordles(word, hw5, limit=9)
        if check_results(sol):
            print(f"{word} solves *ILLS++ with {', '.join(sorted(f'{n} : {c}' for n, c in Counter([c[1] for c in retrieve_all_results(sol)]).items()))}.")
            solvers.append((word,  sol))

'''
# So, it seems that hw2 breaks the hard ones.
>>> for i, word in enumerate([r[0] for r in solvers]):
    ...  sols[word] = compute_all_wordles_hard(word, hw2, words, lookahead=-1)
    ... 
    Solved bgggg:lills on move 7 with guesses: ['vozhd', 'ferny', 'scamp', 'twigs', 'bilks', 'jills'].
    Solved bgggg:zills on move 8 with guesses: ['chevy', 'brand', 'skimp', 'gifts', 'jills', 'lills', 'wills'].
    Solved ggggg:wills on move 7 with guesses: ['chevy', 'brand', 'skimp', 'gifts', 'jills', 'lills', 'wills'].
    Solved ggggg:pills on move 7 with guesses: ['zanja', 'berth', 'comfy', 'disks', 'gills', 'lills', 'pills'].
    Solved bgggg:wills on move 9 with guesses: ['zanja', 'berth', 'comfy', 'disks', 'gills', 'lills', 'pills', 'vills'].
    Solved ggggg:vills on move 8 with guesses: ['zanja', 'berth', 'comfy', 'disks', 'gills', 'lills', 'pills', 'vills'].
    Solved bgggg:zills on move 7 with guesses: ['vughy', 'fremd', 'scant', 'lowps', 'bilks', 'jills'].
    >>> for k, v in sols.items():
        ...  if check_results(v):
            ...   print(k)
            ... 
            >>> [r for r in retrieve_all_results(sols['vughy']) if r[1] > 6]
            [(['vughy', 'fremd', 'scant', 'lowps', 'bilks', 'jills', 'zills'], 7)]
            >>> [r for r in retrieve_all_results(sols['zanja']) if r[1] > 6]
            [(['zanja', 'berth', 'comfy', 'disks', 'gills', 'lills', 'pills'], 7), (['zanja', 'berth', 'comfy', 'disks', 'gills', 'lills', 'pills', 'vills', 'wills'], 9), (['zanja', 'berth', 'comfy', 'disks', 'gills', 'lills', 'pills', 'vills'], 8)]
            >>> [r for r in retrieve_all_results(sols['chevy']) if r[1] > 6]
            [(['chevy', 'brand', 'skimp', 'gifts', 'jills', 'lills', 'wills', 'zills'], 8), (['chevy', 'brand', 'skimp', 'gifts', 'jills', 'lills', 'wills'], 7)]
            >>> [r for r in retrieve_all_results(sols['vozhd']) if r[1] > 6]
            [(['vozhd', 'ferny', 'scamp', 'twigs', 'bilks', 'jills', 'lills'], 7)]
'''
