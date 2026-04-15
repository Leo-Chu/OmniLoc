"""PyTorch Dataset for localization pickles used by main.py."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.feature_extractor import FeatureExtractor


def _to_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _reshape_csi_flat(csi_mag_flat: np.ndarray, n1: int, num_sc: int = 52) -> np.ndarray:
    """Flat [N, n1*num_sc] -> [N, num_sc, n1] for the localization transformer CSI input."""
    n = csi_mag_flat.shape[0]
    rest = csi_mag_flat.reshape(n, n1, num_sc)
    return np.transpose(rest, (0, 2, 1)).astype(np.float32, copy=False)


class LocalizationDataset(Dataset):
    """Loads training.pkl / eval.pkl / test.pkl into model batches.

    Supported on-disk layouts:

    1) Dict with concatenated feature matrix (see ``utils.feature_extractor``):
       - ``features``: [N, 27071] float array
       - Labels via one of:
         - ``labels``: [N, 4]  (x, y, building_id, floor_id)
         - ``label`` / ``targets`` (same)
         - ``building_id``, ``floor_id`` arrays plus ``coords`` or ``xy`` [N, 2]

    2) Dict with per-field arrays (optional):
       - ``Blk_APVec_1``, ``Blk_APVec_2``, ``Blk_CSI_magnitude`` (flat or [N,52,N1]),
         ``Blk_RSSI``, ``Blk_SNR``, and label fields as above.

    3) List of sample dicts with keys ``apvec_1``, ``apvec_2``, ``csi_magnitude``,
       ``rssi``, ``snr``, ``building_id``, ``floor_id``, ``label`` (or x,y in label).
    """

    N1 = 222
    N2 = 3317
    NUM_SC = 52
    # Class indices [0 .. MAX-1]; must match model config (LocalizationLoss CE clamps the same way).
    MAX_BUILDING = 16
    MAX_FLOOR = 10

    def __init__(self, pickle_path: Union[str, Path]):
        super().__init__()
        self.pickle_path = Path(pickle_path)
        if not self.pickle_path.is_file():
            raise FileNotFoundError(f"Pickle not found: {self.pickle_path}")

        with open(self.pickle_path, "rb") as f:
            raw = pickle.load(f)

        self._samples: List[Dict[str, Any]]
        if isinstance(raw, list):
            self._samples = raw
        elif isinstance(raw, dict):
            self._samples = self._dict_to_samples(raw)
        else:
            raise TypeError(f"Expected list or dict in pickle, got {type(raw)}")

        if not self._samples:
            raise ValueError(f"Empty dataset: {self.pickle_path}")

    def _dict_to_samples(self, d: Dict[str, Any]) -> List[Dict[str, Any]]:
        if "features" in d:
            return self._samples_from_concat_features(d)
        req = ("Blk_APVec_1", "Blk_APVec_2", "Blk_RSSI", "Blk_SNR")
        if all(k in d for k in req):
            if "Blk_CSI_magnitude" in d:
                return self._samples_from_field_arrays(d, csi_key="Blk_CSI_magnitude")
            if "Blk_CSI" in d:
                return self._samples_from_field_arrays(d, csi_key="Blk_CSI")
            raise KeyError(
                "Pickle with per-field keys must include Blk_CSI_magnitude or Blk_CSI"
            )
        raise KeyError(
            "Unrecognized pickle dict; need 'features' or Blk_APVec_1/2, "
            "Blk_CSI (or Blk_CSI_magnitude), Blk_RSSI, Blk_SNR"
        )

    def _labels_from_dict(self, d: Dict[str, Any], n: int) -> np.ndarray:
        if "labels" in d:
            y = np.asarray(d["labels"], dtype=np.float32)
        elif "label" in d:
            y = np.asarray(d["label"], dtype=np.float32)
        elif "targets" in d:
            y = np.asarray(d["targets"], dtype=np.float32)
        elif "coords" in d and ("building_id" in d) and ("floor_id" in d):
            xy = np.asarray(d["coords"], dtype=np.float32)
            if xy.ndim == 1:
                xy = xy.reshape(1, -1)
            b = np.asarray(d["building_id"], dtype=np.float32).reshape(-1, 1)
            fl = np.asarray(d["floor_id"], dtype=np.float32).reshape(-1, 1)
            y = np.hstack([xy[:, :2], b, fl])
        else:
            raise KeyError(
                "Need labels as 'labels' [N,4] or coords + building_id + floor_id"
            )
        if y.shape[0] != n:
            raise ValueError(f"Label rows {y.shape[0]} != feature rows {n}")
        if y.shape[1] < 4:
            raise ValueError(f"Labels need at least 4 columns [x,y,bld,flr], got {y.shape}")
        return y

    def _samples_from_concat_features(self, d: Dict[str, Any]) -> List[Dict[str, Any]]:
        features = np.asarray(d["features"], dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"features must be 2D, got {features.shape}")
        n = features.shape[0]
        y = self._labels_from_dict(d, n)
        extractor = FeatureExtractor()
        ap1 = extractor.extract_field(features, "Blk_APVec_1")
        ap2 = extractor.extract_field(features, "Blk_APVec_2")
        csi_flat = extractor.extract_field(features, "Blk_CSI_magnitude")
        rssi = extractor.extract_field(features, "Blk_RSSI")
        snr = extractor.extract_field(features, "Blk_SNR")
        csi = _reshape_csi_flat(csi_flat, self.N1, self.NUM_SC)

        samples: List[Dict[str, Any]] = []
        for i in range(n):
            samples.append(
                {
                    "apvec_1": ap1[i],
                    "apvec_2": ap2[i],
                    "csi_magnitude": csi[i],
                    "rssi": rssi[i],
                    "snr": snr[i],
                    "label": y[i],
                }
            )
        return samples

    def _samples_from_field_arrays(
        self, d: Dict[str, Any], csi_key: str = "Blk_CSI_magnitude"
    ) -> List[Dict[str, Any]]:
        ap1 = np.asarray(d["Blk_APVec_1"], dtype=np.float32)
        ap2 = np.asarray(d["Blk_APVec_2"], dtype=np.float32)
        rssi = np.asarray(d["Blk_RSSI"], dtype=np.float32)
        snr = np.asarray(d["Blk_SNR"], dtype=np.float32)
        csi_raw = np.asarray(d[csi_key], dtype=np.float32)
        n = ap1.shape[0]
        y = self._labels_from_dict(d, n)

        if csi_raw.ndim == 2:
            if csi_raw.shape[1] == self.N1 * self.NUM_SC:
                csi = _reshape_csi_flat(csi_raw, self.N1, self.NUM_SC)
            elif csi_raw.shape[1] == 2 * self.N1 * self.NUM_SC:
                # Legacy concat: magnitude | phase — model uses magnitude only
                mag = csi_raw[:, : self.N1 * self.NUM_SC]
                csi = _reshape_csi_flat(mag, self.N1, self.NUM_SC)
            else:
                raise ValueError(
                    f"{csi_key} 2D shape {csi_raw.shape}; expected "
                    f"[N,{self.N1*self.NUM_SC}] or [N,{2*self.N1*self.NUM_SC}]"
                )
        elif csi_raw.ndim == 3 and csi_raw.shape[1:] == (self.NUM_SC, self.N1):
            csi = csi_raw
        else:
            raise ValueError(
                f"{csi_key} shape {csi_raw.shape}; expected [N,{self.N1*self.NUM_SC}], "
                f"[N,{2*self.N1*self.NUM_SC}], or [N,{self.NUM_SC},{self.N1}]"
            )

        samples: List[Dict[str, Any]] = []
        for i in range(n):
            samples.append(
                {
                    "apvec_1": ap1[i],
                    "apvec_2": ap2[i],
                    "csi_magnitude": csi[i],
                    "rssi": rssi[i],
                    "snr": snr[i],
                    "label": y[i],
                }
            )
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self._samples[idx]
        lab = _to_float32(np.asarray(s["label"]).reshape(-1))[:4].copy()

        def tensor_1d(key: str) -> torch.Tensor:
            return torch.from_numpy(_to_float32(np.asarray(s[key]).reshape(-1)))

        building_id = int(round(float(lab[2])))
        floor_id = int(round(float(lab[3])))
        building_id = max(0, min(self.MAX_BUILDING - 1, building_id))
        floor_id = max(0, min(self.MAX_FLOOR - 1, floor_id))
        lab[2] = float(building_id)
        lab[3] = float(floor_id)

        csi = np.asarray(s["csi_magnitude"], dtype=np.float32)
        if csi.ndim != 2 or csi.shape != (self.NUM_SC, self.N1):
            raise ValueError(
                f"Sample {idx}: csi_magnitude must be [{self.NUM_SC}, {self.N1}], got {csi.shape}"
            )

        return {
            "apvec_1": tensor_1d("apvec_1"),
            "apvec_2": tensor_1d("apvec_2"),
            "csi_magnitude": torch.from_numpy(csi),
            "rssi": tensor_1d("rssi"),
            "snr": tensor_1d("snr"),
            "building_id": torch.tensor(building_id, dtype=torch.long),
            "floor_id": torch.tensor(floor_id, dtype=torch.long),
            "label": torch.from_numpy(lab),
        }
