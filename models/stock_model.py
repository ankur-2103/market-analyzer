from sklearn.ensemble import RandomForestRegressor  # ML model

class StockModel:
    def __init__(self):
        # Create a Random Forest classifier with 100 trees
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    def train(self, X, y):
        """
        Train the model on features X and labels y.
        X: Features (numpy array)
        y: Labels (numpy array)
        """
        self.model.fit(X, y)  # Fit the model to the data
    
    def predict(self, X):
        """
        Predict buy/sell/hold signals for given features X.
        X: Features (numpy array)
        Returns: Predictions (numpy array)
        """
        return self.model.predict(X)  # Predict using the trained model 