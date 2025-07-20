def generate_signals(predictions):
    """
    Convert model predictions to human-readable signals.
    predictions: List or array of model outputs (1, -1, 0)
    Returns: List of strings ('BUY', 'SELL', 'HOLD')
    """
    # Map each prediction to a signal string
    return ["BUY" if p == 1 else "SELL" if p == -1 else "HOLD" for p in predictions] 