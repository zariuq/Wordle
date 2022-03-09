
>>> o = OWordle(sol)
Please play 'swift' for move 1.
>>> o.move('bbbby')
Please play 'lover' for move 2.
'lover'
>>> o.move('bgbbb')
Please play 'tanty' for move 3.
'tanty'
>>> o.move('bbggb')
Please play 'aahed' for move 4.
'aahed'
>>> o.move('bbybb')
You have won with 'month' in 5 moves.
'month'

>>> s = SWordle(swift_solution_skel1)
Please play 'swift' for move 1.
>>> s.move('bbbby')
Please play 'lover' for move 1.
'lover'
>>> s.move('bgbbb')
Please play 'tanty' for move 2.
'tanty'
>>> s.move('bbggb')
Please play 'aahed' for move 3.
'aahed'
>>> s.move('bbybb')
You have won with 'month' in 4 moves.
'month'

>>> s = SWordle(swift_solution2)
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



