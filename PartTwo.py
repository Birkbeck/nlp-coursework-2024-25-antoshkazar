import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report


def load_and_filter_data(path):
    df = pd.read_csv(path)
    df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')
    party_counts = df['party'].value_counts()
    top_4_parties = party_counts.head(4).index.tolist()
    df = df[df['party'].isin(top_4_parties)]
    df = df[df['party'] != 'Speaker']
    df = df[df['speech_class'] == 'Speech']
    df = df[df['speech'].str.len() >= 1000]
    return df


def vectorize_and_split(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
    X = tfidf.fit_transform(df['speech'])
    y = df['party']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=26, stratify=y
    )
    return X_train, X_test, y_train, y_test, tfidf


def vectorize_and_split_uningrams(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 3))
    X = tfidf.fit_transform(df['speech'])
    y = df['party']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=26, stratify=y
    )
    return X_train, X_test, y_train, y_test, tfidf


def train_and_evaluate(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=300, random_state=26)
    svm = SVC(kernel='linear', random_state=26)
    
    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    
    rf_pred = rf.predict(X_test)
    svm_pred = svm.predict(X_test)
    
    print("RandomForest Results:")
    print(f"Macro-average F1 score: {f1_score(y_test, rf_pred, average='macro'):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, rf_pred))
    
    print("\nSVM Results:")
    print(f"Macro-average F1 score: {f1_score(y_test, svm_pred, average='macro'):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, svm_pred))


if __name__ == "__main__":
    #Task 2A
    df = load_and_filter_data('p2-texts/hansard40000.csv')
    print(f"Dataset size after filtering: {df.shape}")
    print(f"Parties in dataset: {df['party'].value_counts().to_dict()}")
    # Task 2B
    X_train, X_test, y_train, y_test, tfidf = vectorize_and_split(df)
    print(f"Train set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    # Task 2C
    train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Task 2D
    X_train, X_test, y_train, y_test, tfidf = vectorize_and_split_uningrams(df)
    print(f"Train set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    train_and_evaluate(X_train, X_test, y_train, y_test)
