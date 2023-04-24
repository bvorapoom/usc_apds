import task1 as cp
import task2 as prep
import sys


def get_transition_default(transition):
    transition_def = {}

    for k, v in transition.items():
        if k[0] != 'start':
            if k[0] not in transition_def.keys():
                transition_def[k[0]] = (k[1], v)
            else:
                if v > transition_def[k[0]][1]:
                    transition_def[k[0]] = (k[1], v)

    return transition_def


def predict_pos_first_word(first_word, transition_init, emission, list_pos, corpus, emission_lower):
    prob_max = 0
    pos_max = None
    if first_word in corpus:
        for pos in list_pos:
            prob_t = transition_init.get(('start', pos), 0)
            prob_e = emission.get((pos, first_word), 0)
            prob = prob_t * prob_e
            if prob > prob_max:
                prob_max = prob
                pos_max = pos
    elif first_word.lower() in list(map(lambda x: x.lower(), corpus)):
        for pos in list_pos:
            prob_t = transition_init.get(('start', pos), 0)
            prob_e = emission_lower.get((pos, first_word.lower()), 0)
            prob = prob_t * prob_e
            if prob > prob_max:
                prob_max = prob
                pos_max = pos

    if pos_max is None:
        # pos_max = max(transition_init, key=transition_init.get)[1]
        pos_max = 'NNP'

    return pos_max



def predict_pos_rest(word, transition, emission, list_pos, corpus, prev_pos, transition_def):
    prob_max = 0
    pos_max = None
    if word in corpus:
        for pos in list_pos:
            prob_t = transition.get((prev_pos, pos), 0)
            prob_e = emission.get((pos, word), 0)
            prob = prob_t * prob_e
            if prob > prob_max:
                prob_max = prob
                pos_max = pos
    if pos_max is None:
        pos_max = transition_def[prev_pos][0]

    return pos_max


def make_prediction(test_data, transition, emission, list_pos, corpus, transition_def, emission_lower, transition_init):
    prediction = list()
    for sentence in test_data:
        temp_prediction = list()
        for word_item in sentence:
            # predict first word
            if word_item[0] == '1':
                pred_pos = predict_pos_first_word(word_item[1], transition_init, emission, list_pos, corpus, emission_lower)
                prev_pos = pred_pos
            # predict the rest
            else:
                pred_pos = predict_pos_rest(word_item[1], transition, emission, list_pos, corpus, prev_pos,
                                            transition_def)
                prev_pos = pred_pos

            temp = (word_item[0], word_item[1], pred_pos)
            temp_prediction.append(temp)
        prediction.append(temp_prediction)

    return prediction


def get_accuracy(test_data, prediction):
    cnt_all = 0
    cnt_match = 0
    for i, sentence in enumerate(test_data):
        for j, word_item in enumerate(sentence):
            cnt_all += 1
            if word_item == prediction[i][j]:
                cnt_match += 1
    return cnt_match / cnt_all


def write_file(pred, path_output):
    output_formatted = '\n\n'.join(['\n'.join(['\t'.join(word) for word in sentence]) for sentence in pred])
    with open(path_output, 'w') as file:
        file.write(output_formatted)


if __name__ == '__main__':
    # input params
    path_train = sys.argv[1]
    path_dev = sys.argv[2]
    path_test = sys.argv[3]
    path_output = sys.argv[4]

    # get data
    train = prep.read_file_to_list(path_train)
    dev = prep.read_file_to_list(path_dev)
    test = prep.read_file_to_list(path_test)

    # get corpus
    corpus = cp.create_corpus(path_train, 0)
    corpus = list(corpus.keys())

    # get transition / emission parameters
    transition, emission, list_pos, emission_lower = prep.get_emission_transition(train)
    transition_init = {k: v for k, v in transition.items() if k[0] == 'start'}
    transition_def = get_transition_default(transition)


    # predict dev
    prediction_dev = make_prediction(dev, transition, emission, list_pos, corpus, transition_def, emission_lower, transition_init)

    # get accuracy
    acc = get_accuracy(dev, prediction_dev)
    print('Greedy Decoding: Accuracy on dev data =', acc)

    # predict test
    prediction_test = make_prediction(test, transition, emission, list_pos, corpus, transition_def, emission_lower, transition_init)
    write_file(prediction_test, path_output)