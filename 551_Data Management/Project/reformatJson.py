import requests
import json
import mysqlupload as m

def reformat_structure(struc):
    if isinstance(struc, str):
        struc = json.loads(struc)

    result = list()
    for k, v in struc.items():
        if isinstance(v, dict):
            if len(v) > 0:
                temp = {'title': k, 'children': reformat_structure(v)}
            else:
                temp = {'title': k}
        elif isinstance(v, list):
            child = [{'title': l} for l in v]
            temp = {'title': k, 'children': child}

        result.append(temp)

    return result

# struc = m.filepath_json()
# print(struc)
# zzz = reformat_structure(struc)
# print(zzz)
#
# print(json.dumps(zzz))