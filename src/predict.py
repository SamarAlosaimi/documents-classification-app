from artifacts import load_artifacts

model, vectorizer = load_artifacts()

def prediction(text):
    X = vectorizer.transform([text])
    pred = model.predict(X)
    return pred[0]

text_1 = "I love Artificial intelligence and machine learning"
text_2 = "Football is my favorite sport"

print(f"Predection of '{text_1}' is: ", prediction(text_1))
print(f"Predection of '{text_2}' is: ", prediction(text_2))