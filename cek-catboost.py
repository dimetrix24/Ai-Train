def check_catboost_device():
    from catboost import CatBoostClassifier
    import numpy as np

    try:
        # Coba fit dummy model di GPU
        model = CatBoostClassifier(
            task_type="GPU",
            devices="0",
            iterations=1,
            verbose=False
        )
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        print("✅ CatBoost GPU is working!")
        return "GPU"
    except Exception as e:
        print("⚠️ CatBoost GPU not available, fallback to CPU.")
        print(f"Reason: {e}")
        return "CPU"