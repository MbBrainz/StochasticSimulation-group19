import csv

def save_to_csv(filedir: str, datalists, headers=[],  mode="w"):
    with open("simulation_data/" + filedir, mode=mode, newline="") as fp:
        writer = csv.writer(fp)
        if not len(headers)==0:
            writer.writerow(headers)

        for i in range(len(datalists[0])):
            row = []
            for i_list in range(len(datalists)):
                row.append(datalists[i_list][i])
            writer.writerow(row)
        print("saved to file!")


import json

def save_results(result_list, filename):
    dict_array = [res.dict() for res in result_list]
    with open(f"simulation_data/{filename}.json", mode="a") as fp:
        json.dump(dict_array, fp)
