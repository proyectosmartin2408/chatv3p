# permite trabajar con el OS
import os

# permite guardar los modelos y manipular los archivos de inteligencia artificial
import pickle

# CountVectorizer convierte texto en un vector
from sklearn.feature_extraction.text import CountVectorizer

# MultinomialNB modelo de inteligencia artificial que aprende relaciones entre texto y respuestas
from sklearn.naive_bayes import MultinomialNB

#  ======================================
# DECLARACION DE VARIABLES GLOBALES

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
ANSWERS_PATH = os.path.join(MODEL_DIR, "answers.pkl")

# ============================================
# Función build_and_train_model
# ============================================
def build_and_train_model(train_pairs):
    questions = [q for q, _ in train_pairs]  # Lista de preguntas
    answers = [a for _, a in train_pairs]  # Lista de respuestas

    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(questions)
    unique_answers = sorted(set(answers))
    answer_to_label = {a: i for i, a in enumerate(unique_answers)}
    y = [answer_to_label[a] for a in answers]
    model = MultinomialNB()
    model.fit(x, y)

    # Crear carpeta
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Guarda objetos entrenados
    with open(MODEL_PATH,"wb") as f:
        pickle.dump(model,f)
    with open(VECTORIZER_PATH,"wb") as f:
        pickle.dump(vectorizer,f)
    with open(ANSWERS_PATH,"wb") as f:
        pickle.dump(unique_answers,f)
    print("OK Modelo entrenado y guardado correctamente")
    return model, vectorizer, unique_answers


#========================================
# FUNCION PARA CARGAR EL MODELO
#========================================
def load_model():
    if (
        os.path.exists(MODEL_PATH) and
        os.path.exists(VECTORIZER_PATH) and
        os.path.exists(ANSWERS_PATH)
    ):
        with open(MODEL_PATH,"rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        with open(ANSWERS_PATH, "rb") as f:
            unique_answers = pickle.load(f)
        print("✔️ Modelo cargado desde disco")
        return model, vectorizer, unique_answers
    else:
        print("⚠️No hay modelo guardado")
        return None, None,None

# =================================================
# Función predict_answers
# =================================================

def predict_answers(model, vectorizer, unique_answers, user_text):
    x = vectorizer.transform([user_text])
    label = model.predict(x)[0]
    return unique_answers[label]