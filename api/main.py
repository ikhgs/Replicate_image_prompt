from flask import Flask, request, jsonify
import replicate
import os
from dotenv import load_dotenv
from collections import defaultdict

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

# Récupérer la clé d'API de Replicate
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

app = Flask(__name__)

# Stocker l'historique des conversations en mémoire
conversation_history = defaultdict(list)

# Configuration de l'API Replicate
def run_replicate(prompt, image_url=None, history=[]):
    # Ajouter l'historique des conversations au prompt
    full_prompt = "\n".join(history + [prompt])
    input_data = {"prompt": full_prompt}
    
    if image_url:
        input_data["image"] = image_url

    output = replicate.run(
        "yorickvp/llava-13b:80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb",
        input=input_data,
        token=REPLICATE_API_TOKEN
    )
    return output

@app.route('/api/lava', methods=['POST'])
def process_request():
    data = request.json
    session_id = data.get('session_id')
    prompt = data.get('prompt')
    image_url = data.get('image_url')  # image_url peut être None ou manquant

    if not session_id or not prompt:
        return jsonify({"error": "Missing session_id or prompt"}), 400

    # Récupérer ou initialiser l'historique des conversations pour cette session
    history = conversation_history[session_id]

    try:
        # Exécuter le modèle avec l'historique des conversations
        response = run_replicate(prompt, image_url, history)

        # Ajouter la réponse au journal des conversations
        history.append(f"User: {prompt}")
        history.append(f"Bot: {response}")

        # Mettre à jour l'historique des conversations
        conversation_history[session_id] = history

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
