import json
from flask import Flask, request, jsonify
import requests
import pmr as p
import pandas as pd
import firebaseUpload as f
import mysqlupload as m
import analytics as a
import mongodbUpload as mg
import reformatJson


app = Flask(__name__)

@app.route('/')
def homepage():
    return app.send_static_file("index.html")


@app.route('/getQueryResult', methods=['GET'])
def get_query_result():
    # query = list(request.args.keys())[0]
    query = request.args['q']
    method = request.args['method']
    source = request.args['source']
    query_result = p.pmr_query_main(method, query, source)
    if isinstance(query_result, pd.DataFrame):
        query_result = query_result.to_string(header=True, index=False).split('\n')
        query_result = '<br>'.join([','.join(ele.split()) for ele in query_result])

    print(query_result)
    # Use this to check the data you send to frontend
    return query_result



@app.route('/terminalCommand', methods=['GET'])
def terminalCommand():
    # query = list(request.args.keys())[0]
    command = request.args['tc']
    source = request.args['source']
    if source == 'firebase':
        firebase_url = 'https://dsci551-project-40a58-default-rtdb.firebaseio.com/'
        tc_result = f.terminal_command(command, firebase_url)
    elif source == 'mysql':
        tc_result = m.terminal_commands(command)
    elif source == 'mongodb':
        tc_result = mg.terminal_command(command)
    print('tc done')

    if isinstance(tc_result, pd.DataFrame):
        tc_result = tc_result.to_string(header=True, index=False).split('\n')
        tc_result = '<br>'.join([','.join(ele.split()) for ele in tc_result])
    elif isinstance(tc_result, list):
        tc_result = '\t'.join(tc_result)

    # print(query_result)
    # Use this to check the data you send to frontend
    return tc_result


@app.route('/analyticsPredict', methods=['GET'])
def analyticsPredict():
    # query = list(request.args.keys())[0]
    algo = request.args['algo']
    # print(request.args)
    # print(algo)

    input_df = pd.DataFrame()
    for attr, val in request.args.items():
        if attr != 'algo' and val != '':
            input_df.loc[0, attr] = val
    print(input_df)
    pred = a.predict(input_df, algo)[0]
    if pred < 0:
        pred = 0
    return str(round(pred, 2))


@app.route('/getStructure', methods=['GET'])
def getStructure():
    # query = list(request.args.keys())[0]
    db = request.args['db']

    if db == 'firebase':
        struc = f.get_structure()
    elif db == 'mysql':
        struc = m.filepath_json()
    elif db == 'mongodb':
        struc = mg.get_structure()

    struc = reformatJson.reformat_structure(struc)
    print(struc)
    return json.dumps(struc)


if __name__ == '__main__':
    app.run(port=3000)