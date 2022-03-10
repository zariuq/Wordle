
Code I used to find some solutions to [Wordle](https://www.nytimes.com/games/wordle/index.html).

The solutions can be found in "solutinos/".  

The notation for hints is 'g' for green hints, 'y' for yellow hints, and 'b' for black (or grey) hints.

The ones with the ending "\*.pp.txt" should be pretty-printed in a manner than humans can parse:

Play "swift".  If the hint is "bbbby", play "carny".  Then if the hint is "bbgby", play "myrrh" and you've won \^o^/.

One caveat is that "myrrh" doesn't enter the official word list.  For more details, please see my blog post, [Adventures in Wordle](https://gardenofminds.art/blog/adventures-in-wordle/), or examine the comments in the code.

---

Included are some Wordle assistant classes:

SWordle parses a skeletal solution (\*.pp.txt) and OWordle parses a 'full' solution (\*.pickle) for you if you want a guaranteed 'win' (assuming new 5-letter words aren't added to the 12,972 currently allowed for guessing).

>>> s = SWordle(swift\_solution2)
Please play 'swift' for move 1.
>>> s.move('bbbby')
Please play 'tater' for move 2.
'tater'
>>> s.move('ybbbb')
Please play 'goony' for move 3.
'goony'
>>> s.move('bgbyb')
You have won with 'month' in 4 moves.
'month'

GWordle performs a greedy search to recommend the next move, so you can start with any word you like.  
GWordle outputs the top 13 moves and the worst-case size of bucket of possible words (but you can examine the score of any word via \*.options or \*.get\_options\_dic()).  
At each stage \*.words allows one to view the possible goals, which enables one to have some poetic fun with the route to victory :>
If a move is not supplied, GWordle assumes you took its top choice.

>>> xo = GWordle(possible\_words=official\_goals)
>>> xo.rec('bbbby', 'swift')
There are 178 possible words.  The best 13 options are:
[(11, 'tater'), (11, 'throe'), (12, 'hoary'), (12, 'potae'), (12, 'rotte'), (12, 'tetra'), (12, 'thrae'), (12, 'troth'), (13, 'haole'), (13, 'harem'), (13, 'hater'), (13, 'lotte'), (13, 'mater')].
'tater'
>>> xo.rec('ybbbb')
There are 10 possible words.  The best 13 options are:
[(1, 'bumph'), (1, 'chump'), (1, 'chums'), (1, 'loony'), (1, 'mouch'), (1, 'mouth'), (1, 'mucho'), (1, 'mulch'), (1, 'mulsh'), (1, 'munch'), (1, 'mutch'), (1, 'rhumb'), (1, 'thumb')].
'bumph'
>>> xo.rec('gbgbg', 'munch')
You will win with 'month' in 4 moves.
'month'

---

words.txt -- 12,972 words that can be used as guesses in Wordle
wordle\_answers\_alphabetical.txt -- 2315 words that were scraped from the website as "official goals"
best\_1k.txt -- 1000 good starting moves
best\_quints\_25.txt -- quintuplets of words that cover 24-25 letters
sgb-words.txt -- 5755 words assembled for a Word Ladder by [Donald Knuth](https://www-cs-faculty.stanford.edu/~knuth/sgb.html).
swiftlover6\_official\_plays.txt -- Simulated plays for each official goal.
swiftlover6\_all\_plays.txt -- Simulated plays for each goal.

words.py -- The main code.  The interactive content is near the end of the file.

