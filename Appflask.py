import tensorflow as tf
import keras
from main import text_pipeline


from flask import Flask, request, render_template

model_path = "Curenetics0.keras"
model = tf.keras.models.load_model(model_path)


test_vectorization = text_pipeline()

application = Flask(__name__)

@application.route('/', methods=["GET", "POST"])
def predict():
    prediction = ""
    score = ""
    if request.method == "POST":
        input_text = request.form["text"]
        vectorized = test_vectorization([input_text])
        pred = model.predict(vectorized)
        label = "Positive" if pred < 0.5 else "Negative"
        prediction = label
        score = round(float(pred), 3)

    return render_template("index.html", prediction=prediction, score=score)



if __name__ == "__main__":
    application.run(debug=True)



