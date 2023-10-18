import ast

def ReadData(filename):
    def extract_data(line):
        result, data_str = line.split(':')
        data = ast.literal_eval(data_str.strip())

        return int(result), data

    with open(filename, 'r') as file:
        lines = file.readlines()

    resultats = []
    donnees = []

    for line in lines:
        result, data = extract_data(line)
        resultats.append([result])  # Ajout sous forme de liste avec un seul élément
        donnees.append(data)

    return donnees, resultats
