from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def evaluate_random_forest(X, Y, **rf_kwargs):
    print("Evaluating with Random Forest")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    rf = RandomForestClassifier(**rf_kwargs)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
