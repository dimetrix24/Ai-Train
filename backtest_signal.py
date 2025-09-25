import os
import pandas as pd
import numpy as np
import logging

from backtester import Backtester


class BacktestWithSignal(Backtester):
    def __init__(self, *args, inverse_mapping=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.inverse_mapping = inverse_mapping
        self.logger.info(f"Inverse mapping: {'ON' if self.inverse_mapping else 'OFF'}")

    def run(self):
        self.logger.info("Starting backtest with signals...")

        # 1. Load data harga
        data = self.load_data()
        if data is None or data.empty:
            self.logger.error("❌ Data harga gagal diload")
            return {"error": "No data"}

        # 2. Feature Engineering
        features = self.apply_feature_engineering(data)
        if features is None or features.empty:
            self.logger.error("❌ Feature engineering gagal")
            return {"error": "No features"}

        self.logger.info(f"Features created: {features.shape}")

        # 3. Prediksi
        probs = self.model.predict_proba(features)
        self.logger.info(f"Predicted probabilities with shape {probs.shape}")

        preds = np.argmax(probs, axis=1)
        signals = pd.Series(preds, index=features.index)

        # 4. Mapping sinyal
        if self.inverse_mapping:
            mapping = {"0": -1, "1": 0, "2": 1}
            signals = signals.astype(str).map(mapping).astype(int)
            self.logger.info(f"Signals mapped. Unique values: {signals.unique().tolist()}")
        else:
            self.logger.warning("⚠️ Inverse mapping dimatikan, prediksi raw dipakai.")
            self.logger.info(f"Signals mapped. Unique values: {signals.unique().tolist()}")

        # Simpan prediksi
        pred_path = os.path.join(self.output_dir, "predictions.csv")
        pd.DataFrame({"signal": signals}).to_csv(pred_path)
        self.logger.info(f"Saved predictions to {pred_path}")

        # 5. Sinkronisasi index dengan data harga
        common_index = data.index.intersection(signals.index)
        if len(common_index) == 0:
            self.logger.error("❌ Tidak ada irisan index antara data harga dan sinyal")
            return {"error": "No common data points"}

        data = data.loc[common_index]
        signals = signals.loc[common_index]

        # 6. Backtest
        results = self.run_backtest(data, signals)
        self.logger.info(f"Final backtest results: {results}")

        # 7. Save results
        if "error" not in results:
            self.save_results(results)

        return results