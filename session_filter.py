# session_filter.py - patched
import pandas as pd
import logging


class SessionFilter:
    def __init__(self, sessions="all", logger=None, skip=None):
        """
        sessions : str | list
            Pilihan sesi: "all", "asia", "europe", "us", "us_late" atau list dari pilihan itu.
        skip : list | None
            Daftar sesi yang ingin di-skip (prioritas lebih tinggi daripada sessions).
        """
        if isinstance(sessions, str):
            self.sessions = [sessions.lower()]
        else:
            self.sessions = [s.lower() for s in sessions]

        self.skip = [s.lower() for s in skip] if skip else []
        self.logger = logger or logging.getLogger(__name__)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataset sesuai sesi trading.
        Asumsi: df.index sudah DatetimeIndex (UTC/GMT).
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame harus pakai DatetimeIndex")

        if "all" in self.sessions and not self.skip:
            self.logger.info("No session filter applied (using all data).")
            return df

        df = df.copy()
        df["hour"] = df.index.hour

        # mapping sesi ke range jam
        session_map = {
            "asia": (0, 8),       # 00:00–08:00 UTC
            "europe": (7, 15),    # 07:00–15:00 UTC
            "us": (13, 21),       # 13:00–21:00 UTC
            "us_late": (18, 21),  # 18:00–21:00 UTC
        }

        if self.skip:
            # Buang data dari sesi yang di-skip
            mask = pd.Series(True, index=df.index)
            for s in self.skip:
                if s not in session_map:
                    raise ValueError(f"Sesi tidak dikenali (skip): {s}")
                start, end = session_map[s]
                mask &= ~((df["hour"] >= start) & (df["hour"] < end))
            filtered = df.loc[mask]
            mode = f"skip={self.skip}"
        else:
            # Ambil hanya sesi yang dipilih
            mask = pd.Series(False, index=df.index)
            for s in self.sessions:
                if s == "all":
                    mask |= True
                elif s in session_map:
                    start, end = session_map[s]
                    mask |= (df["hour"] >= start) & (df["hour"] < end)
                else:
                    raise ValueError(f"Sesi tidak dikenali: {s}")
            filtered = df.loc[mask]
            mode = f"keep={self.sessions}"

        filtered = filtered.drop(columns=["hour"])
        self.logger.info(f"Applied session filter: {mode} | Rows: {len(filtered)} / {len(df)}")
        return filtered