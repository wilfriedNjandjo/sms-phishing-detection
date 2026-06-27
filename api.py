from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re

app = Flask(__name__)
CORS(app)

modele = joblib.load("models/modele_final.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")

def nettoyer_texte(texte):
    texte = str(texte).lower()
    texte = re.sub(r"http\S+|www\.\S+", " urltoken ", texte)
    texte = re.sub(r"\b\d{5,}\b", " numtoken ", texte)
    texte = re.sub(r"[^\w\s]", " ", texte)
    texte = re.sub(r"\d+", " num ", texte)
    texte = re.sub(r"\s+", " ", texte).strip()
    return texte

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sms = data.get("sms", "")
    if sms.strip() == "":
        return jsonify({"erreur": "le sms est vide"}), 400

    sms_clean = nettoyer_texte(sms)
    vecteur = vectorizer.transform([sms_clean])
    pred = modele.predict(vecteur)[0]
    label = "Arnaque" if pred == 1 else "Legitime"
    score = modele.decision_function(vecteur)[0]

    return jsonify({"sms": sms, "label": label, "score_decision": round(float(score), 4)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
