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
    dict_prob = dict()
    if first_word in corpus:
        for pos in list_pos:
            if (pos, first_word) in emission.keys():
                prob_t = transition_init.get(pos, (0, 0))[0]
                prob_e = emission.get((pos, first_word), 0)
                prob = prob_t * prob_e
                dict_prob[pos] = (prob, [pos])
    # elif first_word.lower() in list(map(lambda x: x.lower(), corpus)):
    #     for pos in list_pos:
    #         if (pos, first_word.lower()) in emission_lower.keys():
    #             prob_t = transition_init.get(('start', pos), 0)
    #             prob_e = emission_lower.get((pos, first_word.lower()), 0)
    #             prob = prob_t * prob_e
    #             dict_prob[pos] = (prob, [pos])

    if len(dict_prob) == 0:
        dict_prob = transition_init

    return dict_prob


def predict_pos_rest(word, transition, emission, list_pos, corpus, transition_def, dict_prob_prev):
    dict_prob = dict()
    if word in corpus:
        for pos in list_pos:
            if (pos, word) in emission.keys():
                prob_max = -1e5
                list_pos = None
                for _, (prev_prob, prev_pos_list) in dict_prob_prev.items():
                    prev_pos = prev_pos_list[-1]
                    prob_t = transition.get((prev_pos, pos), 0)
                    prob_e = emission.get((pos, word), 0)
                    prob = prob_t * prob_e * prev_prob
                    if prob > prob_max:
                        prob_max = prob
                        list_pos = prev_pos_list + [pos]
                dict_prob[pos] = (prob_max, list_pos)

    if len(dict_prob) == 0:
        for _, (prev_prob, prev_pos_list) in dict_prob_prev.items():
            if prev_pos_list[-1] in transition_def.keys():
                pos = transition_def[prev_pos_list[-1]][0]
                prob_t = transition_def[prev_pos_list[-1]][1]
                prob = prev_prob * prob_t
                # print(prev_pos_list, pos, word)
                list_pos = prev_pos_list + [pos]
                dict_prob[pos] = (prob, list_pos)

    return dict_prob


def make_prediction(test_data, transition, emission, list_pos, corpus, transition_def, emission_lower, transition_init):
    prediction = list()
    for sentence in test_data:
        for word_item in sentence:
            # predict first word
            if word_item[0] == '1':
                dict_prob = predict_pos_first_word(word_item[1], transition_init, emission, list_pos, corpus, emission_lower)

            # predict the rest
            else:
                dict_prob = predict_pos_rest(word_item[1], transition, emission, list_pos, corpus, transition_def, dict_prob)

        prob_max = -1e5
        list_prediction = None
        for _, (prob, pred) in dict_prob.items():
            if prob > prob_max:
                prob_max = prob
                list_prediction = pred

        temp_prediction = [(word_item[0], word_item[1], list_prediction[i]) for i, word_item in enumerate(sentence)]
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
    transition_init = {k[1]: (v, [k[1]]) for k, v in transition.items() if k[0] == 'start'}
    transition_def = get_transition_default(transition)


    # predict
    prediction_dev = make_prediction(dev, transition, emission, list_pos, corpus, transition_def, emission_lower, transition_init)

    # get accuracy
    acc = get_accuracy(dev, prediction_dev)
    print('Viterbi Decoding: Accuracy on dev data =', acc)

    # predict test
    prediction_test = make_prediction(test, transition, emission, list_pos, corpus, transition_def, emission_lower, transition_init)
    write_file(prediction_test, path_output)