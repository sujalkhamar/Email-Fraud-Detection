import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from src.config import RANDOM_STATE, CV_FOLDS, MODEL_SAVE_PATH, PROCESSED_DATA_PATH
from src.logger import setup_logger

logger = setup_logger(__name__)

class ModelTrainer:
    def __init__(self):
        self.models = {
            "LogisticRegression": {
                "model": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
                "params": {"C": [0.1, 1, 10]}
            },
            "MultinomialNB": {
                "model": MultinomialNB(),
                "params": {"alpha": [0.1, 1.0, 10.0]}
            },
            "LinearSVC": {
                "model": LinearSVC(random_state=RANDOM_STATE, max_iter=2000),
                "params": {"C": [0.1, 1, 10]}
            }
        }
        self.best_model = None
        self.best_score = 0

    def train(self):
        """Trains multiple models and selects the best one based on F1-score."""
        logger.info("Loading processed text data.")
        data = joblib.load(PROCESSED_DATA_PATH)
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]

        results = []

        for name, config in self.models.items():
            logger.info(f"Training {name} with Grid Search...")
            grid = GridSearchCV(
                config["model"], 
                config["params"], 
                cv=CV_FOLDS, 
                scoring='f1', 
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            
            model = grid.best_estimator_
            y_pred = model.predict(X_test)
            
            # Metrics
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            # LinearSVC doesn't have predict_proba by default
            if hasattr(model, "predict_proba"):
                roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, model.decision_function(X_test))
            
            logger.info(f"{name} -> F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, ROC-AUC: {roc_auc:.4f}")
            
            results.append({
                "model_name": name,
                "model": model,
                "f1": f1
            })

            if f1 > self.best_score:
                self.best_score = f1
                self.best_model = model
                self.best_model_name = name

        logger.info(f"Best Model: {self.best_model_name} with F1-score: {self.best_score:.4f}")
        
        # Save best model
        joblib.dump(self.best_model, MODEL_SAVE_PATH)
        logger.info(f"Best model saved to {MODEL_SAVE_PATH}")
        
        return self.best_model
