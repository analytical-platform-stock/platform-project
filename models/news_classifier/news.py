import pandas as pd
from nltk.tokenize import word_tokenize
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from catboost import CatBoostClassifier
import optuna
import pickle

nltk.download('punkt', force=True)
nlp = spacy.load('ru_core_news_sm')


def load_and_preprocess_data(filepath, ticket_filter):
    data = pd.read_csv(filepath, sep=';')
    data = data[data['ticket'] == ticket_filter]
    data = data.dropna()
    data['tokens'] = data['announce'].fillna("").apply(word_tokenize)
    data['tokens'] = data['tokens'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
    return data


def process_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_punct])


def vectorize_text(data):
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X = tfidf_vectorizer.fit_transform(data['tokens'])
    return X, data['labels'], tfidf_vectorizer


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_catboost(X_train, y_train, params=None):
    default_params = {'iterations': 1000, 'learning_rate': 0.1, 'depth': 6, 'loss_function': 'MultiClass', 'verbose': 0}
    if params:
        default_params.update(params)
    model = CatBoostClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    return accuracy, roc_auc


def optimize_catboost(X_train, X_test, y_train, y_test):
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 6, 12),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 128),
            'loss_function': 'MultiClass',
            'verbose': 0
        }
        model = train_catboost(X_train, y_train, params)
        _, roc_auc = evaluate_model(model, X_test, y_test)
        return roc_auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    return study.best_params


def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def main():
    data = load_and_preprocess_data('D:\\analytic_platform_api\\news\\bcs_express_news.csv', 'ROSN')
    X, y, tfidf_vectorizer = vectorize_text(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr_model = train_logistic_regression(X_train, y_train)
    acc_lr, roc_auc_lr = evaluate_model(lr_model, X_test, y_test)
    print(f"Logistic Regression - Accuracy: {acc_lr:.4f}, ROC AUC: {roc_auc_lr:.4f}")

    best_params = optimize_catboost(X_train, X_test, y_train, y_test)
    catboost_model = train_catboost(X_train, y_train, best_params)
    acc_cb, roc_auc_cb = evaluate_model(catboost_model, X_test, y_test)
    print(f"CatBoost - Accuracy: {acc_cb:.4f}, ROC AUC: {roc_auc_cb:.4f}")

    save_model(catboost_model, 'catboost_model.pkl')


if __name__ == "__main__":
    main()