from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Charger le modèle
with open("model(tc).pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Bienvenue dans l'API de régression !"

@app.route("/predict", methods=["POST"])
def predict():
    d20=float(input)
    
    test = {

    'd20(Kg/m3)': [791.6, 794.3],
    'n20': [1.444, 1.4454],
    'Tb(K)': [629.7, 641.8],
    'MM(g/mole)': [296.580353, 310.607239 ],
    'famille': ["n-paraffines","n-paraffines"],
    'nbH': [44,46],
    'nbC': [21,22]

}

# Création du DataFrame
test = pd.DataFrame(test)

if __name__ == "__main__":
    app.run(debug=True)
