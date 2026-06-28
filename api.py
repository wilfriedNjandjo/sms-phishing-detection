from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import logging
import datetime

app = Flask(__name__)
CORS(app)

# Configuration du logging : tout s'affiche dans les logs Render automatiquement
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("monitoring")

modele = joblib.load("models/modele_final.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")

# Compteurs simples gardés en mémoire pendant que le service tourne
# (remis à zéro si le service redémarre ou se met en veille - limite connue du tier gratuit)
compteurs = {"total": 0, "arnaque": 0, "legitime": 0}

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

    # --- MONITORING : mise a jour des compteurs et ecriture dans les logs ---
    compteurs["total"] += 1
    if label == "Arnaque":
        compteurs["arnaque"] += 1
    else:
        compteurs["legitime"] += 1

    taux_arnaque = compteurs["arnaque"] / compteurs["total"]

    # on n'enregistre jamais le contenu exact du SMS (vie privee), juste sa longueur et le resultat
    logger.info(
        f"[PREDICTION] horodatage={datetime.datetime.now().isoformat()} "
        f"longueur_sms={len(sms)} label={label} score={round(float(score),4)} "
        f"total={compteurs['total']} taux_arnaque={round(taux_arnaque,3)}"
    )

    # alerte simple si le taux d'arnaques detectees devient anormalement eleve
    if compteurs["total"] >= 10 and taux_arnaque > 0.70:
        logger.warning(
            f"[ALERTE] taux d'arnaques anormalement eleve : {round(taux_arnaque,3)} "
            f"sur {compteurs['total']} predictions"
        )

    return jsonify({"sms": sms, "label": label, "score_decision": round(float(score), 4)})

@app.route("/stats", methods=["GET"])
def stats():
    """Route de monitoring : renvoie les statistiques cumulees depuis le dernier demarrage du service."""
    total = compteurs["total"]
    taux_arnaque = (compteurs["arnaque"] / total) if total > 0 else 0
    return jsonify({
        "total_predictions": total,
        "nb_arnaque": compteurs["arnaque"],
        "nb_legitime": compteurs["legitime"],
        "taux_arnaque": round(taux_arnaque, 3)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
