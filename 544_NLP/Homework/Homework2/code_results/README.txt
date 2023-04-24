- Python 3.9 is used to run the code
- there are 4 .py files: task1.py, task2.py, task3.py, and task4.py for each of the 4 tasks of assignmenet 2 accordingly

HOW TO RUN EACH .py FILE
## run in terminal with arguments

task1.py
terminal command format: python3 task1.py <path_train> <threshold> <path_output>
example: python3 task1.py 'data/train' 0 'vocab.txt'

task2.py
terminal command format: python3 task2.py <path_train> <path_output>
example: python3 task2.py 'data/train' 'hmm.json'

task3.py
terminal command format: python3 task3.py <path_train> <path_dev> <path_test> <path_output>
example: python3 task3.py 'data/train' 'data/dev' 'data/test' 'greedy.out'

task4.py
terminal command format: python3 task4.py <path_train> <path_dev> <path_test> <path_output>
example: python3 task4.py 'data/train' 'data/dev' 'data/test' 'viterbi.out'