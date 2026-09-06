# Это патч для experiment.py — вставляем исправленную функцию
import numpy as np

def get_mnist_data_fixed(n_samples=500, seed=42):
    from sklearn.datasets import fetch_openml
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    print("Loading MNIST...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='pandas')
    # Приводим y к int и берём только 0 и 1
    y = y.astype(int)
    mask = (y == 0) | (y == 1)
    X = X[mask]
    y = y[mask]
    print(f"Found {len(y)} samples of digits 0 and 1")
    if len(y) == 0:
        raise RuntimeError("No samples with digits 0 or 1 found. Check MNIST data.")
    # Уменьшаем до n_samples
    rng = np.random.RandomState(seed)
    if len(y) > n_samples:
        indices = rng.choice(len(y), n_samples, replace=False)
        X = X[indices]
        y = y[indices]
    # Стандартизация и PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=20)
    X_reduced = pca.fit_transform(X_scaled)
    print(f"MNIST loaded: {len(X_reduced)} samples, {X_reduced.shape[1]} features")
    return X_reduced, y
