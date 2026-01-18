from sklearn.feature_extraction.text import CountVectorizer

def split_features_labels(df):
    X = df["text"]
    y = df["label"]
    return X, y

def build_vectorizer():
    return CountVectorizer()

def fit_vectorizer(vectorizer, X_train):
    return vectorizer.fit_transform(X_train)

def transform_text(vectorizer, X_test):
    return vectorizer.transform(X_test)

