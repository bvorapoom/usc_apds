There are 4 .py files in total
- task1.py --> for training task1
- task1_test.py --> for getting predictions using pre-trained model task1
- task2.py --> for training task2
- task2_test.py --> for getting predictions using pre-trained model task2




task1.py
- Use the following command line to run the code: 
python3 task1.py <path_train> <path_dev> <path_test>
- This will generate 4 files in total: dev1.out (prediction for dev data), test1.out (prediction for test data), blstm1.pt (model), and merge_dev_1.txt (use to check performance on conll03eval file)
- training time can take up to 30 minutes


task1_test.py
- Use the following command line to run the code: 
python3 task1.py <path_train> <path_dev> <path_test> <path_model)
- This will generate 3 files in total: dev1.out (prediction for dev data), test1.out (prediction for test data), and merge_dev_1.txt (use to check performance on conll03eval file)


task2.py
- Use the following command line to run the code: 
python3 task2.py <path_train> <path_dev> <path_test>
- This will generate 4 files in total: dev2.out (prediction for dev data), test2.out (prediction for test data), blstm2.pt (model), and merge_dev_2.txt (use to check performance on conll03eval file)
- training time can take up to 30 minutes


task2_test.py
- Use the following command line to run the code: 
python3 task1.py <path_train> <path_dev> <path_test> <path_model)
- This will generate 3 files in total: dev2.out (prediction for dev data), test2.out (prediction for test data), and merge_dev_2.txt (use to check performance on conll03eval file)