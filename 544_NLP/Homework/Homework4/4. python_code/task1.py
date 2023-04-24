import numpy as np
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import copy
from sklearn.metrics import classification_report
import warnings
import sys


def read_file_to_list(path_input):
    output_list = list()
    with open(path_input, 'r') as file:
        data = file.read()
        for sentence in data.split('\n\n'):
            format_sentence = [tuple(line.split()) for line in sentence.strip('\n').split('\n')]
            output_list.append(format_sentence)
    return output_list


def create_corpus(train_list):
    dict_word2idx = dict()
    for sentence in train_list:
        for _, word, _ in sentence:
            if word not in dict_word2idx.keys():
                dict_word2idx[word] = len(dict_word2idx) + 1

    return dict_word2idx


def create_corpus_mask(train_list):
    dict_word2idx = dict()
    for sentence in train_list:
        for _, word, _ in sentence:
            if word.lower() not in dict_word2idx.keys():
                dict_word2idx[word.lower()] = len(dict_word2idx) + 1

    return dict_word2idx


def prepare_data(data, dict_word2idx, dict_ner2idx, true_label=True):
    # convert raw data (tuple) to index of each word in the sentence using the dict_word2idx created
    # pad sequence

    # max_len = max([len(x) for x in data])
    max_len = 20

    if true_label:
        output_x, output_y = list(), list()
        for sentence in data:
            temp_x, temp_y = list(), list()
            for word_item in sentence:
                word = word_item[1]
                ner = word_item[2]

                if word in dict_word2idx.keys():
                    temp_x.append(dict_word2idx[word])
                else:
                    temp_x.append(0)

                if ner in dict_ner2idx.keys():
                    temp_y.append(dict_ner2idx[ner])
                else:
                    temp_y.append(-1)

            temp_x = (temp_x + max_len * [0])[:max_len]
            temp_y = (temp_y + max_len * [-1])[:max_len]
            output_x.append(temp_x)
            output_y.append(temp_y)
        return np.array(output_x), np.array(output_y)
    else:
        output_x = list()
        for sentence in data:
            temp = list()
            for word_item in sentence:
                word = word_item[1]
                if word in dict_word2idx.keys():
                    temp.append(dict_word2idx[word])
                else:
                    temp.append(0)

            temp = (temp + max_len * [0])[:max_len]
            output_x.append(temp)
        return np.array(output_x)


def has_numbers(inputString):
    return int(any(char.isdigit() for char in inputString))


def prepare_data_mask(data, dict_word2idx, dict_ner2idx, true_label=True):
    # convert raw data (tuple) to index of each word in the sentence using the dict_word2idx created
    # pad sequence

    # max_len = max([len(x) for x in data])
    max_len = 35

    empty_mask = [0, 0, 0, 0]

    if true_label:
        output_x, output_y, output_mask = list(), list(), list()
        for sentence in data:
            temp_x, temp_y, temp_mask = list(), list(), list()
            for word_item in sentence:
                word_ind = word_item[0]
                word = word_item[1]
                ner = word_item[2]

                mask_capital = int(word[0].isupper())
                mask_start = int(word_ind == '1')
                mask_hasNum = has_numbers(word)
                mask_allUpper = int(word.isupper())

                mask = np.array([mask_capital, mask_start, mask_hasNum, mask_allUpper])
                temp_mask.append(mask)

                if word in dict_word2idx.keys():
                    temp_x.append(dict_word2idx[word])
                    if ner in dict_ner2idx.keys():
                        temp_y.append(dict_ner2idx[ner])
                    else:
                        temp_y.append(-1)
                else:
                    temp_x.append(0)
                    temp_y.append(-1)



            temp_x = (temp_x + max_len * [0])[:max_len]
            temp_mask = (temp_mask + [empty_mask for i in range(max_len)])[:max_len]
            temp_y = (temp_y + max_len * [-1])[:max_len]
            output_x.append(temp_x)
            output_y.append(temp_y)
            output_mask.append(temp_mask)
        return np.array(output_x), np.array(output_y), np.array(output_mask)
    else:
        output_x, output_mask = list(), list()
        for sentence in data:
            temp_x, temp_mask = list(), list()
            for word_item in sentence:
                word_ind = word_item[0]
                word = word_item[1]

                mask_capital = int(word[0].isupper())
                mask_start = int(word_ind == '1')
                mask_hasNum = has_numbers(word)
                mask_allUpper = int(word.isupper())

                mask = np.array([mask_capital, mask_start, mask_hasNum, mask_allUpper])
                temp_mask.append(mask)

                if word in dict_word2idx.keys():
                    temp_x.append(dict_word2idx[word])
                else:
                    temp_x.append(0)

            temp_x = (temp_x + max_len * [0])[:max_len]
            temp_mask = (temp_mask + [empty_mask for i in range(max_len)])[:max_len]
            output_x.append(temp_x)
            output_mask.append(temp_mask)
        return np.array(output_x), np.array(output_mask)


class model_BiLSTM(nn.Module):
    def __init__(self, vocab_size, num_class):
        super(model_BiLSTM, self).__init__()

        # Defining some parameters
        self.n_dim_embedding = 100
        self.n_dim_hidden_lstm = 256
        self.dropout_lstm = 0.33
        self.n_dim_output_linear = 128

        # Defining the layers
        self.embedding = nn.Embedding(vocab_size, self.n_dim_embedding)
        self.lstm = nn.LSTM(input_size=self.n_dim_embedding+4, hidden_size=self.n_dim_hidden_lstm, \
                            batch_first=True, bias=True, bidirectional=True, dropout=self.dropout_lstm)
        self.linear = nn.Linear(self.n_dim_hidden_lstm * 2, self.n_dim_output_linear)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(self.n_dim_output_linear, num_class)

    def forward(self, x, mask):
        # Initializing hidden state for first input using method defined below

        # embedding
        embeds = self.embedding(x)

        # add mask
        embeds = torch.cat((embeds, mask), dim=-1)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, _ = self.lstm(embeds)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.linear(out)
        out = self.elu(out)
        out = self.classifier(out)

        return out


def train(n_epochs, model, train_loader, valid_loader, optimizer, criterion, ts_t_X, ts_t_y, ts_v_X, ts_v_y, mask_t, mask_v, print_every=1):
    # initialize tracker for minimum validation loss
    best_f1 = -np.Inf
    best_sd = None

    for epoch in range(n_epochs):
        # monitor losses
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training

        for data, mask, target in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data, mask)
            # calculate the loss
            loss = criterion(output.view(-1, 9), target.view(-1))
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        with torch.no_grad():
            for data, mask, target in valid_loader:
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data, mask)
                # calculate the loss\
                loss = criterion(output.view(-1, 9), target.view(-1))
                # update running validation loss
                valid_loss += loss.item()

        train_acc, _, _, train_f1, len_train = report_scores(model(ts_t_X, mask_t), ts_t_y)
        valid_acc, _, _, valid_f1, len_valid = report_scores(model(ts_v_X, mask_v), ts_v_y)

        train_loss = train_loss / len_train * 10
        valid_loss = valid_loss / len_valid * 10

        # save params of the best model
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            best_sd = copy.deepcopy(model.state_dict())

        if epoch % print_every == 0:
            print(
                'Epoch: {}\tTrain Loss: {:.3f} || Valid Loss: {:.3f} || Train Acc: {:.3f} || Valid Acc: {:.3f} || Valid F1: {:.3f}'.format(
                    epoch + 1, train_loss, valid_loss, train_acc, valid_acc, valid_f1))

    return best_sd


def report_scores(y_pred, y_true):
    prediction = torch.argmax(y_pred, dim=-1)
    mask = y_true > -1
    prediction = prediction[mask]

    # accuracy
    num_match = (prediction == y_true[mask]).sum()
    num_total = prediction.shape[0]
    acc = (num_match / num_total).item()

    # precision, recall, f1
    dict_report = classification_report(y_true[mask], prediction, output_dict=True)

    cum_prec, cum_rec, cnt = 0, 0, 0
    for i, val in dict_report.items():
        if i not in ['0', 'accuracy', 'micro avg', 'weighted avg']:
            cnt += val['support']
            cum_prec += val['precision'] * val['support']
            cum_rec = val['recall'] * val['support']

    prec = cum_prec / cnt
    rec = cum_rec / cnt
    f1 = 2 * prec * rec / (prec + rec)

    return acc, prec, rec, f1, num_total


def format_prediction(raw_list, y_pred, dict_ind2ner):
    y_pred_np = y_pred.numpy()
    y_pred_format = list()
    for ind_s, sentence in enumerate(raw_list[:len(y_pred)]):
        temp = list()
        for ind_w, word_item in enumerate(sentence):
            index = word_item[0]
            word = word_item[1]
            try:
                pred = dict_ind2ner[y_pred_np[ind_s][ind_w]]
            except:
                pred = 'O'
            tup = (index, word, pred)
            temp.append(tup)
        y_pred_format.append(temp)
    return y_pred_format


def write_file(pred, path_output):
    output_formatted = '\n\n'.join(['\n'.join([' '.join(word) for word in sentence]) for sentence in pred])
    with open(path_output, 'w') as file:
        file.write(output_formatted)


def merge_truth_pred(raw_list, y_pred_format):
    merge = list()
    for ind_s, sentence in enumerate(y_pred_format):
        temp = list()
        for ind_w, word_item in enumerate(sentence):
            index = word_item[0]
            word = word_item[1]
            pred = word_item[2]
            truth = raw_list[ind_s][ind_w][2]
            tup = (index, word, truth, pred)
            temp.append(tup)
        merge.append(temp)
    return merge


def save_model(model, path):
    torch.save(model, path)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # path_train = sys.argv[1]
    # path_dev = sys.argv[2]
    # path_test = sys.argv[3]

    path_train = 'data/train'
    path_dev = 'data/dev'
    path_test = 'data/test'

    # read all data into list of lists
    train_list = read_file_to_list(path_train)
    dev_list = read_file_to_list(path_dev)
    test_list = read_file_to_list(path_test)

    # create corpus from training data / index mapping
    dict_word2idx = create_corpus(train_list)
    dict_ner2idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4,
                    'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    dict_ind2ner = {v: k for k, v in dict_ner2idx.items()}


    # prepare data (map with index, padding)
    X_train, y_train, mask_train = prepare_data_mask(train_list, dict_word2idx, dict_ner2idx)
    X_dev, y_dev, mask_dev = prepare_data_mask(dev_list, dict_word2idx, dict_ner2idx)
    X_test, mask_test = prepare_data_mask(test_list, dict_word2idx, dict_ner2idx, true_label=False)


    # calculate weights of classes to be used in nn.CrossEntropyLoss
    weights = dict()
    for sent in y_train:
        for lab in sent:
            if lab != -1:
                weights[lab] = weights.get(lab, 0) + 1
    weights = [i for _, i in sorted(weights.items())]
    weights = torch.tensor(weights) / sum(weights)
    weights = 1. / weights
    # weights[0] = 0.01

    # to run sample code
    # X_train, y_train = X_train[:100], y_train[:100]
    # X_dev, y_dev = X_dev[:100], y_dev[:100]
    # mask_train = mask_train[:100]
    # mask_dev = mask_dev[:100]


    # convert data to tensors & dataloaders
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    mask_train_tensor = torch.from_numpy(mask_train)
    train_tensor = data_utils.TensorDataset(X_train_tensor, mask_train_tensor, y_train_tensor)
    train_loader = data_utils.DataLoader(train_tensor, batch_size=10, shuffle=True)

    X_dev_tensor = torch.from_numpy(X_dev)
    y_dev_tensor = torch.from_numpy(y_dev)
    mask_dev_tensor = torch.from_numpy(mask_dev)
    dev_tensor = data_utils.TensorDataset(X_dev_tensor, mask_dev_tensor, y_dev_tensor)
    dev_loader = data_utils.DataLoader(dev_tensor, batch_size=10, shuffle=True)

    X_test_tensor = torch.from_numpy(X_test)
    mask_test_tensor = torch.from_numpy(mask_test)

    # initialize the NN
    model = model_BiLSTM(len(dict_word2idx)+1, len(dict_ner2idx))

    # specify loss function (categorical cross-entropy)
    # criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=weights, reduction='mean')
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

    # specify optimizer (stochastic gradient descent) and learning rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    # train BiLSTM
    print('-----------training BiLSTM-----------')
    best_sd = train(20, model, train_loader, dev_loader, optimizer, criterion, \
                       X_train_tensor, y_train_tensor, X_dev_tensor, y_dev_tensor, mask_train_tensor, mask_dev_tensor, \
                       print_every=1)

    # report scores of best model
    model_best = model_BiLSTM(len(dict_word2idx)+1, len(dict_ner2idx))
    model_best.load_state_dict(best_sd)

    train_acc, train_prec, train_rec, train_f1, _ = report_scores(model_best(X_train_tensor, mask_train_tensor), y_train_tensor)
    dev_acc, dev_prec, dev_rec, dev_f1, _ = report_scores(model_best(X_dev_tensor, mask_dev_tensor), y_dev_tensor)

    print('-----------best model\'s scores-----------')
    print('Training:')
    print('Accuracy = {:.4f}, Precision = {:.4f}, Recall = {:.4f}, F1-score = {:.4f}'.format(
        train_acc, train_prec, train_rec, train_f1))
    print('Testing:')
    print('Accuracy = {:.4f}, Precision = {:.4f}, Recall = {:.4f}, F1-score = {:.4f}'.format(
        dev_acc, dev_prec, dev_rec, dev_f1))


    # get predictions
    print('-----------getting predictions-----------')
    y_pred_train = torch.argmax(model_best(X_train_tensor, mask_train_tensor), dim=-1)
    y_pred_dev = torch.argmax(model_best(X_dev_tensor, mask_dev_tensor), dim=-1)
    y_pred_test = torch.argmax(model_best(X_test_tensor, mask_test_tensor), dim=-1)

    y_pred_train_format = format_prediction(train_list, y_pred_train, dict_ind2ner)
    y_pred_dev_format = format_prediction(dev_list, y_pred_dev, dict_ind2ner)
    y_pred_test_format = format_prediction(test_list, y_pred_test, dict_ind2ner)

    merge_pred_train = merge_truth_pred(train_list, y_pred_train_format)
    merge_pred = merge_truth_pred(dev_list, y_pred_dev_format)

    # write files
    write_file(y_pred_dev_format, '3-imp/dev1.out')
    write_file(y_pred_test_format, '3-imp/test1.out')
    write_file(merge_pred, '3-imp/merge_dev_1.txt')

    # save models
    save_model(model_best, '3-imp/blstm1.pt')


