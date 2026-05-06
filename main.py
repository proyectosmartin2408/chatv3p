from flask import Flask, render_template, request, jsonify
from chatbot.data import training_data
from chatbot.model import build_and_train_model, load_model, predict_answers

app = Flask(__name__)

model, vectorizer, unique_answers = load_model()

if model is None:
    model, vectorizer, unique_answers = build_and_train_model(training_data)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/america")
def america():
    return render_template("america.html")


@app.route("/europa")
def europa():
    return render_template("europa.html")


@app.route("/asia")
def asia():
    return render_template("asia.html")


@app.route("/africa")
def africa():
    return render_template("africa.html")


@app.route("/oceania")
def oceania():
    return render_template("oceania.html")


@app.route("/chat", methods=["POST"])

# --------------------------------------------
def chat():
    user_text = request.form.get("message","")
    if not user_text.strip():
        return jsonify({"response":"por favor escribe algo"})
    
    try:
        # Asegúrate de que model y vectorizer existan
        response = predict_answers(model, vectorizer, unique_answers, user_text)
        return jsonify({"response": response})
    
    except Exception as e:
        # Manejo de errores para evitar que el server se caiga
        print(f"Error en la predicción: {e}")
        return jsonify({"response": "Lo siento, hubo un error interno."}), 500

# -------------------  funcion principal
if __name__ == "__main__":  
    app.run(debug=True)                                                
    app.run(host="0.0.0.0", port=5000)
