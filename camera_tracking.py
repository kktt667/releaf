from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
import hashlib
import json
import logging
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

import cv2
from flask import Flask, jsonify, request
import numpy as np
from PIL import Image
import qrcode


@dataclass
class TrackingSignals:
    looking_score: float
    vegetation_score: float
    outdoor_score: float
    grass_outside_score: float
    looking_at_screen: bool
    vegetation_detected: bool
    outdoor_detected: bool
    grass_outside_detected: bool
    face_detected: bool
    eyes_detected: bool


@dataclass
class UploadInference:
    vegetation_score: float
    outdoor_score: float
    grass_outside_score: float
    vegetation_detected: bool
    outdoor_detected: bool
    grass_outside_detected: bool


@dataclass
class UploadState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    latest_frame: np.ndarray | None = None
    upload_count: int = 0
    latest_status: str = "No upload yet"
    latest_session_id: str = ""
    latest_received_at: float = 0.0
    active_session_id: str = ""
    session_upload_counts: dict[str, int] = field(default_factory=dict)
    latest_token_id: str = ""
    latest_proof_url: str = ""
    latest_flower_level: int = 1
    latest_value: float = 0.0
    latest_block_hash: str = ""
    latest_chain_badge: str = "CHAIN_FALLBACK_LOCAL_LEDGER"


class RecoveryAppState(str, Enum):
    ONLINE = "ONLINE"
    WARNING = "WARNING"
    DECAY = "DECAY"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    OUTSIDE_MODE = "OUTSIDE_MODE"
    PROOF_VERIFIED = "PROOF_VERIFIED"
    MINTED = "MINTED"


class ChainBadge(str, Enum):
    CHAIN_LIVE = "CHAIN_LIVE"
    CHAIN_RETRYING = "CHAIN_RETRYING"
    CHAIN_FALLBACK_LOCAL_LEDGER = "CHAIN_FALLBACK_LOCAL_LEDGER"


@dataclass
class RecoverySession:
    session_id: str
    started_at: float
    state_started_at: float
    user_present_at_start: bool
    left_screen_at: float | None = None
    absence_validated_at: float | None = None
    upload_count: int = 0
    proof_status: str = "pending"
    vegetation_score: float = 0.0
    outdoor_score: float = 0.0
    grass_outside_score: float = 0.0
    nft_tx_hash: str = ""
    token_id: str = ""
    proof_url: str = ""
    metadata_path: str = ""
    completed_at: float | None = None


@dataclass
class ChainActionResult:
    badge: ChainBadge
    token_id: str = ""
    tx_hash: str = ""
    error: str = ""
    proof_url: str = ""
    metadata_path: str = ""
    local_block_hash: str = ""


@dataclass
class LiveMetrics:
    outdoor_sessions: int = 0
    streak_days: int = 0
    decay_penalties: int = 0
    score_value: float = 0.0
    nft_owner: str = ""
    token_id: str = ""
    latest_tx_hash: str = ""
    last_outdoor_day: str = ""


class RecoveryLedger:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data = self._load_or_create()

    def _default_data(self) -> dict:
        return {
            "sessions": [],
            "metrics": {
                "outdoor_sessions": 0,
                "streak_days": 0,
                "decay_penalties": 0,
                "score_value": 0.0,
                "nft_owner": "",
                "token_id": "",
                "latest_tx_hash": "",
                "last_outdoor_day": "",
            },
            "last_chain_result": {
                "badge": ChainBadge.CHAIN_FALLBACK_LOCAL_LEDGER.value,
                "token_id": "",
                "tx_hash": "",
                "proof_url": "",
                "metadata_path": "",
                "local_block_hash": "",
            },
        }

    def _load_or_create(self) -> dict:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.isfile(self.path):
            data = self._default_data()
            self._write(data)
            return data
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if not isinstance(loaded, dict):
                raise ValueError("ledger root must be dict")
            return loaded
        except Exception:
            data = self._default_data()
            self._write(data)
            return data

    def _write(self, payload: dict) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

    def metrics(self) -> LiveMetrics:
        with self._lock:
            m = self._data.get("metrics", {})
            return LiveMetrics(
                outdoor_sessions=int(m.get("outdoor_sessions", 0)),
                streak_days=int(m.get("streak_days", 0)),
                decay_penalties=int(m.get("decay_penalties", 0)),
                score_value=float(m.get("score_value", 0.0)),
                nft_owner=str(m.get("nft_owner", "")),
                token_id=str(m.get("token_id", "")),
                latest_tx_hash=str(m.get("latest_tx_hash", "")),
                last_outdoor_day=str(m.get("last_outdoor_day", "")),
            )

    def mark_decay(self) -> LiveMetrics:
        with self._lock:
            m = self._data["metrics"]
            m["decay_penalties"] = int(m.get("decay_penalties", 0)) + 1
            self._recompute_value_unlocked()
            self._write(self._data)
        return self.metrics()

    def record_verified_session(
        self,
        session: RecoverySession,
        wallet_address: str,
        chain_result: ChainActionResult,
    ) -> LiveMetrics:
        with self._lock:
            today = time.strftime("%Y-%m-%d")
            m = self._data["metrics"]
            prev_day = str(m.get("last_outdoor_day", ""))
            if prev_day == today:
                pass
            else:
                if prev_day:
                    prev_epoch = time.mktime(time.strptime(prev_day, "%Y-%m-%d"))
                    today_epoch = time.mktime(time.strptime(today, "%Y-%m-%d"))
                    if (today_epoch - prev_epoch) <= 86400.0 * 1.5:
                        m["streak_days"] = int(m.get("streak_days", 0)) + 1
                    else:
                        m["streak_days"] = 1
                else:
                    m["streak_days"] = 1
                m["last_outdoor_day"] = today
                m["outdoor_sessions"] = int(m.get("outdoor_sessions", 0)) + 1

            m["nft_owner"] = wallet_address
            if chain_result.token_id:
                m["token_id"] = chain_result.token_id
            if chain_result.tx_hash:
                m["latest_tx_hash"] = chain_result.tx_hash
            self._data["last_chain_result"] = {
                "badge": chain_result.badge.value,
                "token_id": chain_result.token_id,
                "tx_hash": chain_result.tx_hash,
                "error": chain_result.error,
                "proof_url": chain_result.proof_url,
                "metadata_path": chain_result.metadata_path,
                "local_block_hash": chain_result.local_block_hash,
            }

            self._data["sessions"].append(
                {
                    "session_id": session.session_id,
                    "started_at": session.started_at,
                    "completed_at": session.completed_at,
                    "proof_status": session.proof_status,
                    "vegetation_score": session.vegetation_score,
                    "outdoor_score": session.outdoor_score,
                    "grass_outside_score": session.grass_outside_score,
                    "token_id": session.token_id,
                    "tx_hash": session.nft_tx_hash,
                    "proof_url": session.proof_url,
                    "metadata_path": session.metadata_path,
                    "local_block_hash": chain_result.local_block_hash,
                }
            )
            self._recompute_value_unlocked()
            self._write(self._data)
        return self.metrics()

    def _recompute_value_unlocked(self) -> None:
        m = self._data["metrics"]
        base = 1.0
        value = (
            base
            + (0.5 * float(m.get("outdoor_sessions", 0)))
            + (1.0 * float(m.get("streak_days", 0)))
            - (0.3 * float(m.get("decay_penalties", 0)))
        )
        m["score_value"] = float(max(0.0, round(value, 2)))


class ThirdwebBridge:
    def __init__(
        self,
        enabled: bool,
        mint_url: str,
        update_url: str,
        api_key: str,
        timeout_seconds: float = 12.0,
    ) -> None:
        self.enabled = enabled
        self.mint_url = mint_url.strip()
        self.update_url = update_url.strip()
        self.api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds

    def mint_or_update(
        self,
        wallet_address: str,
        metadata: dict,
        existing_token_id: str,
    ) -> ChainActionResult:
        if not self.enabled:
            return ChainActionResult(
                badge=ChainBadge.CHAIN_FALLBACK_LOCAL_LEDGER,
                token_id=existing_token_id,
            )
        target_url = self.update_url if existing_token_id else self.mint_url
        if not target_url:
            return ChainActionResult(
                badge=ChainBadge.CHAIN_FALLBACK_LOCAL_LEDGER,
                token_id=existing_token_id,
                error="missing_chain_endpoint",
            )
        payload = {
            "wallet": wallet_address,
            "token_id": existing_token_id,
            "metadata": metadata,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            target_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
            parsed = json.loads(raw) if raw else {}
            return ChainActionResult(
                badge=ChainBadge.CHAIN_LIVE,
                token_id=str(parsed.get("token_id", existing_token_id)),
                tx_hash=str(parsed.get("tx_hash", "")),
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return ChainActionResult(
                badge=ChainBadge.CHAIN_RETRYING,
                token_id=existing_token_id,
                error=str(exc),
            )


class LocalImmutableChain:
    """Append-only local chain simulator with hash-linked blocks."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def _last_block_unlocked(self) -> dict | None:
        if not os.path.isfile(self.path):
            return None
        last: dict | None = None
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        last = parsed
                except json.JSONDecodeError:
                    continue
        return last

    def append_event(self, event: dict) -> tuple[int, str]:
        with self._lock:
            prev = self._last_block_unlocked()
            index = (int(prev.get("index", -1)) + 1) if prev else 0
            prev_hash = str(prev.get("block_hash", "GENESIS")) if prev else "GENESIS"
            block = {
                "index": index,
                "ts": time.time(),
                "prev_hash": prev_hash,
                "event": event,
            }
            block_body = json.dumps(block, sort_keys=True, separators=(",", ":"))
            block_hash = hashlib.sha256(block_body.encode("utf-8")).hexdigest()
            block["block_hash"] = block_hash
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(block, sort_keys=True) + "\n")
            return index, block_hash


class CameraTracker:
    def __init__(
        self,
        looking_threshold: float = 0.62,
        vegetation_threshold: float = 0.24,
        outdoor_threshold: float = 0.35,
        smoothing: float = 0.85,
    ) -> None:
        self.looking_threshold = looking_threshold
        self.vegetation_threshold = vegetation_threshold
        self.outdoor_threshold = outdoor_threshold
        self.smoothing = smoothing

        self._look_ema = 0.0
        self._veg_ema = 0.0
        self._out_ema = 0.0
        self._last_face_cx: float | None = None
        self._last_face_cy: float | None = None

        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        self._eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )
        self._ml_pos_samples: list[np.ndarray] = []
        self._ml_neg_samples: list[np.ndarray] = []
        self._out_pos_samples: list[np.ndarray] = []
        self._out_neg_samples: list[np.ndarray] = []

    def process(self, frame_bgr: np.ndarray) -> TrackingSignals:
        look_score, face_detected, eyes_detected = self._compute_looking_score(frame_bgr)
        vegetation_score = self._predict_vegetation_score(frame_bgr)
        outdoor_score = self._predict_outdoor_score(frame_bgr)

        self._look_ema = self._ema(self._look_ema, look_score)
        self._veg_ema = self._ema(self._veg_ema, vegetation_score)
        self._out_ema = self._ema(self._out_ema, outdoor_score)
        grass_outside_score = self._veg_ema * self._out_ema

        return TrackingSignals(
            looking_score=self._look_ema,
            vegetation_score=self._veg_ema,
            outdoor_score=self._out_ema,
            grass_outside_score=grass_outside_score,
            looking_at_screen=self._look_ema >= self.looking_threshold,
            vegetation_detected=self._veg_ema >= self.vegetation_threshold,
            outdoor_detected=self._out_ema >= self.outdoor_threshold,
            grass_outside_detected=grass_outside_score >= (self.vegetation_threshold * self.outdoor_threshold),
            face_detected=face_detected,
            eyes_detected=eyes_detected,
        )

    def process_looking_only(self, frame_bgr: np.ndarray) -> TrackingSignals:
        look_score, face_detected, eyes_detected = self._compute_looking_score(frame_bgr)
        self._look_ema = self._ema(self._look_ema, look_score)
        return TrackingSignals(
            looking_score=self._look_ema,
            vegetation_score=0.0,
            outdoor_score=0.0,
            grass_outside_score=0.0,
            looking_at_screen=self._look_ema >= self.looking_threshold,
            vegetation_detected=False,
            outdoor_detected=False,
            grass_outside_detected=False,
            face_detected=face_detected,
            eyes_detected=eyes_detected,
        )

    def infer_upload(self, frame_bgr: np.ndarray) -> UploadInference:
        vegetation_score = self._predict_vegetation_score(frame_bgr)
        outdoor_score = self._predict_outdoor_score(frame_bgr)
        grass_outside_score = vegetation_score * outdoor_score
        vegetation_detected = vegetation_score >= self.vegetation_threshold
        outdoor_detected = outdoor_score >= self.outdoor_threshold
        grass_outside_detected = grass_outside_score >= (self.vegetation_threshold * self.outdoor_threshold)
        return UploadInference(
            vegetation_score=vegetation_score,
            outdoor_score=outdoor_score,
            grass_outside_score=grass_outside_score,
            vegetation_detected=vegetation_detected,
            outdoor_detected=outdoor_detected,
            grass_outside_detected=grass_outside_detected,
        )

    def _detect_faces(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        frontal = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(90, 90),
        )
        if len(frontal) > 0:
            return list(frontal)

        # Fallback profile detection for side-turned faces.
        left_profiles = self._profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(90, 90),
        )
        flipped = cv2.flip(gray, 1)
        right_profiles = self._profile_cascade.detectMultiScale(
            flipped,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(90, 90),
        )
        frame_w = gray.shape[1]
        right_converted = [
            (frame_w - (x + w), y, w, h) for (x, y, w, h) in right_profiles
        ]
        return list(left_profiles) + right_converted

    def _compute_looking_score(self, frame_bgr: np.ndarray) -> tuple[float, bool, bool]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._detect_faces(gray)
        if len(faces) == 0:
            self._last_face_cx = None
            self._last_face_cy = None
            return 0.0, False, False

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        frame_h, frame_w = gray.shape
        face_cx = (x + (w / 2.0)) / frame_w
        face_cy = (y + (h / 2.0)) / frame_h

        center_error = abs(face_cx - 0.5) + abs(face_cy - 0.44)
        centered_score = float(np.clip(1.0 - (1.9 * center_error), 0.0, 1.0))

        face_ratio = (w * h) / float(frame_w * frame_h)
        size_score = float(np.clip((face_ratio - 0.025) / 0.11, 0.0, 1.0))

        face_roi = gray[y : y + h, x : x + w]
        eyes = self._eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(18, 18),
        )
        if len(eyes) >= 2:
            eyes_score = 1.0
        elif len(eyes) == 1:
            eyes_score = 0.5
        else:
            eyes_score = 0.1

        if self._last_face_cx is None or self._last_face_cy is None:
            stability_score = 0.5
        else:
            movement = abs(face_cx - self._last_face_cx) + abs(face_cy - self._last_face_cy)
            stability_score = float(np.clip(1.0 - 5.0 * movement, 0.0, 1.0))

        self._last_face_cx = face_cx
        self._last_face_cy = face_cy

        looking_score = (
            0.45 * centered_score
            + 0.20 * size_score
            + 0.25 * eyes_score
            + 0.10 * stability_score
        )
        return float(np.clip(looking_score, 0.0, 1.0)), True, len(eyes) >= 1

    def _compute_vegetation_score(self, frame_bgr: np.ndarray) -> float:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        # Green mask for foliage-like tones.
        green_mask = (
            (h >= 28) & (h <= 92) &
            (s >= 45) &
            (v >= 35)
        )
        green_ratio = float(np.count_nonzero(green_mask)) / float(green_mask.size)
        if green_ratio < 0.04:
            return 0.0

        # Uniform green fabric often has low hue variance.
        green_hues = h[green_mask].astype(np.float32)
        hue_std = float(np.std(green_hues)) if green_hues.size > 0 else 0.0
        hue_variation_score = float(np.clip(hue_std / 18.0, 0.0, 1.0))

        # Plants/grass usually have fine edges and non-uniform texture.
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edge_on_green = float(np.count_nonzero((edges > 0) & green_mask))
        green_pixels = float(np.count_nonzero(green_mask))
        edge_density = edge_on_green / max(green_pixels, 1.0)
        texture_score = float(np.clip(edge_density / 0.18, 0.0, 1.0))

        # Penalize one huge flat region (typical of a shirt filling frame).
        mask_u8 = green_mask.astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if num_labels <= 1:
            largest_component_ratio = 1.0
            component_score = 0.0
        else:
            areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
            largest_component_ratio = float(np.max(areas) / max(green_pixels, 1.0))
            component_score = float(np.clip((len(areas) - 1) / 6.0, 0.0, 1.0))

        green_ratio_score = float(np.clip((green_ratio - 0.03) / 0.35, 0.0, 1.0))
        flat_region_penalty = float(np.clip((largest_component_ratio - 0.75) / 0.25, 0.0, 1.0))

        vegetation_score = (
            0.30 * green_ratio_score
            + 0.28 * hue_variation_score
            + 0.27 * texture_score
            + 0.15 * component_score
            - 0.20 * flat_region_penalty
        )
        return float(np.clip(vegetation_score, 0.0, 1.0))

    def _compute_outdoor_heuristic_score(self, frame_bgr: np.ndarray) -> float:
        resized = cv2.resize(frame_bgr, (200, 140), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        top = resized[: resized.shape[0] // 3, :, :]
        top_hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
        top_h = top_hsv[:, :, 0]
        top_s = top_hsv[:, :, 1]
        top_v = top_hsv[:, :, 2]

        sky_mask = (
            (top_h >= 85) & (top_h <= 130) &
            (top_s >= 25) & (top_s <= 190) &
            (top_v >= 80)
        )
        sky_ratio = float(np.count_nonzero(sky_mask)) / float(sky_mask.size)
        sky_score = float(np.clip(sky_ratio / 0.18, 0.0, 1.0))

        brightness = float(np.mean(v) / 255.0)
        brightness_score = float(np.clip((brightness - 0.30) / 0.45, 0.0, 1.0))

        sat = float(np.mean(s) / 255.0)
        sat_score = float(np.clip((sat - 0.10) / 0.35, 0.0, 1.0))

        h_std = float(np.std(h.astype(np.float32)) / 90.0)
        color_var_score = float(np.clip(h_std, 0.0, 1.0))

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edge_density = float(np.mean(cv2.Canny(gray, 80, 160) > 0))
        edge_score = float(np.clip((edge_density - 0.02) / 0.12, 0.0, 1.0))

        outdoor = (
            0.35 * sky_score
            + 0.22 * brightness_score
            + 0.18 * sat_score
            + 0.15 * color_var_score
            + 0.10 * edge_score
        )
        return float(np.clip(outdoor, 0.0, 1.0))

    def _extract_ml_features(self, frame_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame_bgr, (160, 120), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
        hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        hist = hist / (np.sum(hist) + 1e-6)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edge_density = np.array([np.mean(edges > 0)], dtype=np.float32)

        h = hsv[:, :, 0].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)
        stats = np.array(
            [
                np.mean(h) / 180.0,
                np.std(h) / 90.0,
                np.mean(s) / 255.0,
                np.std(s) / 128.0,
                np.mean(v) / 255.0,
                np.std(v) / 128.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([hist, edge_density, stats]).astype(np.float32)

    def _extract_outdoor_features(self, frame_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame_bgr, (200, 140), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)

        top = hsv[: hsv.shape[0] // 3, :, :]
        top_h = top[:, :, 0].astype(np.float32)
        top_s = top[:, :, 1].astype(np.float32)
        top_v = top[:, :, 2].astype(np.float32)
        sky_mask = (
            (top_h >= 85) & (top_h <= 130) &
            (top_s >= 25) & (top_s <= 190) &
            (top_v >= 80)
        )
        sky_ratio = np.array([np.mean(sky_mask)], dtype=np.float32)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edge_density = np.array([np.mean(edges > 0)], dtype=np.float32)

        # Color histograms for general scene context.
        h_hist = cv2.calcHist([hsv], [0], None, [10], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
        hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        hist = hist / (np.sum(hist) + 1e-6)

        stats = np.array(
            [
                np.mean(h) / 180.0,
                np.std(h) / 90.0,
                np.mean(s) / 255.0,
                np.std(s) / 128.0,
                np.mean(v) / 255.0,
                np.std(v) / 128.0,
                np.mean(top_v) / 255.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([hist, edge_density, sky_ratio, stats]).astype(np.float32)

    def add_ml_sample(self, frame_bgr: np.ndarray, is_vegetation: bool) -> None:
        feat = self._extract_ml_features(frame_bgr)
        if is_vegetation:
            self._ml_pos_samples.append(feat)
            if len(self._ml_pos_samples) > 80:
                self._ml_pos_samples.pop(0)
        else:
            self._ml_neg_samples.append(feat)
            if len(self._ml_neg_samples) > 80:
                self._ml_neg_samples.pop(0)

    def add_outdoor_sample(self, frame_bgr: np.ndarray, is_outdoor: bool) -> None:
        feat = self._extract_outdoor_features(frame_bgr)
        if is_outdoor:
            self._out_pos_samples.append(feat)
            if len(self._out_pos_samples) > 80:
                self._out_pos_samples.pop(0)
        else:
            self._out_neg_samples.append(feat)
            if len(self._out_neg_samples) > 80:
                self._out_neg_samples.pop(0)

    def ml_sample_counts(self) -> tuple[int, int, int, int]:
        return (
            len(self._ml_pos_samples),
            len(self._ml_neg_samples),
            len(self._out_pos_samples),
            len(self._out_neg_samples),
        )

    def _knn_prob(
        self,
        query: np.ndarray,
        pos_samples: list[np.ndarray],
        neg_samples: list[np.ndarray],
    ) -> float | None:
        if len(pos_samples) < 4 or len(neg_samples) < 4:
            return None
        pos = np.stack(pos_samples, axis=0)
        neg = np.stack(neg_samples, axis=0)
        data = np.concatenate([pos, neg], axis=0)
        labels = np.concatenate(
            [np.ones(pos.shape[0], dtype=np.float32), np.zeros(neg.shape[0], dtype=np.float32)]
        )
        dists = np.linalg.norm(data - query[None, :], axis=1)
        k = min(7, len(dists))
        idx = np.argpartition(dists, k - 1)[:k]
        chosen_labels = labels[idx]
        chosen_dists = dists[idx]
        weights = 1.0 / (chosen_dists + 1e-6)
        weighted_prob = float(np.sum(chosen_labels * weights) / np.sum(weights))
        return float(np.clip(weighted_prob, 0.0, 1.0))

    def _compute_ml_vegetation_score(self, frame_bgr: np.ndarray) -> float | None:
        query = self._extract_ml_features(frame_bgr)
        return self._knn_prob(query, self._ml_pos_samples, self._ml_neg_samples)

    def _compute_outdoor_ml_score(self, frame_bgr: np.ndarray) -> float | None:
        query = self._extract_outdoor_features(frame_bgr)
        return self._knn_prob(query, self._out_pos_samples, self._out_neg_samples)

    def _compute_outdoor_score(self, frame_bgr: np.ndarray) -> float:
        heur = self._compute_outdoor_heuristic_score(frame_bgr)
        ml = self._compute_outdoor_ml_score(frame_bgr)
        if ml is None:
            return heur
        return float(np.clip(0.45 * heur + 0.55 * ml, 0.0, 1.0))

    def _predict_vegetation_score(self, frame_bgr: np.ndarray) -> float:
        vegetation_score = self._compute_vegetation_score(frame_bgr)
        ml_score = self._compute_ml_vegetation_score(frame_bgr)
        if ml_score is not None:
            vegetation_score = 0.35 * vegetation_score + 0.65 * ml_score
        return float(np.clip(vegetation_score, 0.0, 1.0))

    def _predict_outdoor_score(self, frame_bgr: np.ndarray) -> float:
        return self._compute_outdoor_score(frame_bgr)

    def _read_training_image(self, path: str) -> np.ndarray | None:
        frame = cv2.imread(path)
        if frame is not None:
            return frame
        try:
            pil_img = Image.open(path).convert("RGB")
            rgb = np.array(pil_img)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def bootstrap_training_from_folder(self, folder_path: str) -> dict[str, int]:
        stats = {
            "files_seen": 0,
            "files_loaded": 0,
            "decode_failed": 0,
            "veg_pos": 0,
            "veg_neg": 0,
            "out_pos": 0,
            "out_neg": 0,
        }
        if not os.path.isdir(folder_path):
            return stats

        valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}
        inside_dir = os.path.join(folder_path, "inside")
        outside_dir = os.path.join(folder_path, "outside")
        outside_grass_dir = os.path.join(outside_dir, "grass")
        outside_not_grass_dir = os.path.join(outside_dir, "not grass")
        outside_not_grass_alt = os.path.join(outside_dir, "not_grass")

        def add_labeled_dir(
            dir_path: str,
            vegetation_label: bool | None,
            outdoor_label: bool | None,
        ) -> None:
            if not os.path.isdir(dir_path):
                return
            for name in sorted(os.listdir(dir_path)):
                path = os.path.join(dir_path, name)
                if not os.path.isfile(path):
                    continue
                _, ext = os.path.splitext(name.lower())
                if ext not in valid_ext:
                    continue
                stats["files_seen"] += 1

                frame = self._read_training_image(path)
                if frame is None:
                    stats["decode_failed"] += 1
                    continue

                stats["files_loaded"] += 1
                if vegetation_label is True:
                    self.add_ml_sample(frame, is_vegetation=True)
                    stats["veg_pos"] += 1
                elif vegetation_label is False:
                    self.add_ml_sample(frame, is_vegetation=False)
                    stats["veg_neg"] += 1

                if outdoor_label is True:
                    self.add_outdoor_sample(frame, is_outdoor=True)
                    stats["out_pos"] += 1
                elif outdoor_label is False:
                    self.add_outdoor_sample(frame, is_outdoor=False)
                    stats["out_neg"] += 1

        labeled_mode = (
            os.path.isdir(inside_dir)
            or os.path.isdir(outside_grass_dir)
            or os.path.isdir(outside_not_grass_dir)
            or os.path.isdir(outside_not_grass_alt)
        )

        if labeled_mode:
            # User-provided labels:
            # inside/*      => indoor + not grass
            # outside/grass => outdoor + grass
            # outside/not grass|not_grass => outdoor + not grass
            add_labeled_dir(inside_dir, vegetation_label=False, outdoor_label=False)
            add_labeled_dir(outside_grass_dir, vegetation_label=True, outdoor_label=True)
            add_labeled_dir(outside_not_grass_dir, vegetation_label=False, outdoor_label=True)
            add_labeled_dir(outside_not_grass_alt, vegetation_label=False, outdoor_label=True)
            return stats

        for name in sorted(os.listdir(folder_path)):
            path = os.path.join(folder_path, name)
            if not os.path.isfile(path):
                continue
            _, ext = os.path.splitext(name.lower())
            if ext not in valid_ext:
                continue
            stats["files_seen"] += 1

            frame = self._read_training_image(path)
            if frame is None:
                stats["decode_failed"] += 1
                continue

            stats["files_loaded"] += 1
            veg_h = self._compute_vegetation_score(frame)
            out_h = self._compute_outdoor_heuristic_score(frame)

            if veg_h >= 0.45:
                self.add_ml_sample(frame, is_vegetation=True)
                stats["veg_pos"] += 1
            elif veg_h <= 0.12:
                self.add_ml_sample(frame, is_vegetation=False)
                stats["veg_neg"] += 1

            if out_h >= 0.50:
                self.add_outdoor_sample(frame, is_outdoor=True)
                stats["out_pos"] += 1
            elif out_h <= 0.18:
                self.add_outdoor_sample(frame, is_outdoor=False)
                stats["out_neg"] += 1

        return stats

    def _ema(self, prev: float, current: float) -> float:
        a = self.smoothing
        return a * prev + (1.0 - a) * current


def open_camera_with_retry(camera_index: int, timeout_seconds: float) -> cv2.VideoCapture | None:
    cam: cv2.VideoCapture | None = None
    deadline = time.time() + max(timeout_seconds, 0.5)
    backends: list[int | None]
    if sys.platform == "darwin":
        backends = [cv2.CAP_AVFOUNDATION, None]
    else:
        backends = [None]

    while time.time() < deadline:
        if cam is not None:
            cam.release()
        for backend in backends:
            if backend is None:
                cam = cv2.VideoCapture(camera_index)
            else:
                cam = cv2.VideoCapture(camera_index, backend)
            if cam.isOpened():
                return cam
            cam.release()
        time.sleep(0.35)
    if cam is not None:
        cam.release()
    return None


def open_camera_source_with_retry(source: str, timeout_seconds: float) -> cv2.VideoCapture | None:
    cam: cv2.VideoCapture | None = None
    deadline = time.time() + max(timeout_seconds, 0.5)
    while time.time() < deadline:
        if cam is not None:
            cam.release()
        cam = cv2.VideoCapture(source)
        if cam.isOpened():
            return cam
        time.sleep(0.35)
    if cam is not None:
        cam.release()
    return None


def get_local_ip() -> str:
    candidates: list[str] = []

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        candidates.append(sock.getsockname()[0])
    except OSError:
        pass
    finally:
        sock.close()

    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        for info in infos:
            ip = info[4][0]
            if ip:
                candidates.append(ip)
    except OSError:
        pass

    # Prefer private LAN addresses that phones can reach on same Wi-Fi.
    for ip in candidates:
        if ip.startswith("192.168.") or ip.startswith("10.") or ip.startswith("172."):
            return ip
    for ip in candidates:
        if ip and not ip.startswith("127."):
            return ip
    return "127.0.0.1"


def make_qr_tile(url: str, size_px: int = 180) -> np.ndarray:
    qr_img = qrcode.make(url).convert("RGB")
    qr_np = np.array(qr_img)
    qr_np = cv2.resize(qr_np, (size_px, size_px), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(qr_np, cv2.COLOR_RGB2BGR)


def _safe_resize_keep_aspect(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img
    scale = max(target_h, 1) / float(h)
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(target_h))), interpolation=cv2.INTER_AREA)


def _load_flower_assets(flowers_folder: str) -> list[np.ndarray]:
    if not os.path.isdir(flowers_folder):
        return []
    assets: list[np.ndarray] = []
    for name in sorted(os.listdir(flowers_folder)):
        lower = name.lower()
        if not (lower.endswith(".png") or lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".webp")):
            continue
        path = os.path.join(flowers_folder, name)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            assets.append(img)
    return assets


def _draw_flower_fallback(canvas: np.ndarray, x: int, y: int, level: int) -> None:
    stem_h = 48 + min(level, 8) * 4
    cv2.line(canvas, (x, y), (x, y - stem_h), (40, 180, 60), 4, cv2.LINE_AA)
    petal_color = (70 + min(level * 15, 160), 80, 220)
    center = (x, y - stem_h)
    for dx, dy in ((-15, 0), (15, 0), (0, -15), (0, 15), (-11, -11), (11, -11)):
        cv2.circle(canvas, (center[0] + dx, center[1] + dy), 11, petal_color, -1, cv2.LINE_AA)
    cv2.circle(canvas, center, 10, (30, 190, 230), -1, cv2.LINE_AA)


def _draw_flower_asset(canvas: np.ndarray, flower_assets: list[np.ndarray], level: int, x: int, y: int, target_h: int) -> None:
    if not flower_assets:
        _draw_flower_fallback(canvas, x + 36, y + target_h - 8, level)
        return
    idx = min(max(level - 1, 0), len(flower_assets) - 1)
    asset = _safe_resize_keep_aspect(flower_assets[idx], target_h=target_h)
    h, w = asset.shape[:2]
    if y + h > canvas.shape[0] or x + w > canvas.shape[1]:
        return
    if len(asset.shape) < 3:
        rgb = cv2.cvtColor(asset, cv2.COLOR_GRAY2BGR)
        canvas[y : y + h, x : x + w] = rgb
    elif asset.shape[2] == 4:
        alpha = (asset[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
        base = canvas[y : y + h, x : x + w].astype(np.float32)
        src = asset[:, :, :3].astype(np.float32)
        mixed = (alpha * src) + ((1.0 - alpha) * base)
        canvas[y : y + h, x : x + w] = mixed.astype(np.uint8)
    else:
        canvas[y : y + h, x : x + w] = asset[:, :, :3]


def _build_terminal_upload_page(session_id: str) -> str:
    page = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>SYS::UPLOAD</title>
  <link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=VT323&display=swap" rel="stylesheet"/>
  <style>
    :root {
      --green: #00ff41;
      --green-dim: #00aa2b;
      --green-glow: rgba(0,255,65,0.4);
      --bg: #020a02;
      --surface: #040f04;
      --border: #00ff4133;
      --red: #ff003c;
      --amber: #ffb300;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--green);
      font-family: 'Share Tech Mono', monospace;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      position: relative;
    }
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.15) 2px, rgba(0,0,0,0.15) 4px);
      pointer-events: none;
      z-index: 100;
    }
    body::after {
      content: '';
      position: fixed;
      inset: 0;
      background: radial-gradient(ellipse at center, transparent 60%, rgba(0,0,0,0.85) 100%);
      pointer-events: none;
      z-index: 99;
    }
    #rain { position: fixed; inset: 0; opacity: 0.07; z-index: 0; }
    .terminal {
      position: relative;
      z-index: 10;
      width: min(600px, 95vw);
      border: 1px solid var(--green-dim);
      background: var(--surface);
      box-shadow: 0 0 40px var(--green-glow), inset 0 0 60px rgba(0,0,0,0.5);
      animation: flicker 8s infinite;
    }
    @keyframes flicker { 0%,95%,100%{opacity:1;} 96%{opacity:0.92;} 97%{opacity:1;} 98%{opacity:0.85;} 99%{opacity:1;} }
    .term-header {
      background: #001a00;
      border-bottom: 1px solid var(--green-dim);
      padding: 8px 16px;
      display: flex;
      align-items: center;
      gap: 12px;
      font-family: 'VT323', monospace;
      font-size: 1.1rem;
      letter-spacing: 0.1em;
    }
    .term-dots { display: flex; gap: 6px; }
    .dot { width: 10px; height: 10px; border-radius: 50%; border: 1px solid currentColor; }
    .dot.r { color: var(--red); background: var(--red); }
    .dot.a { color: var(--amber); background: var(--amber); }
    .dot.g { color: var(--green); background: var(--green); }
    .term-title { flex: 1; text-align: center; color: var(--green-dim); text-shadow: 0 0 8px var(--green-glow); }
    .term-body { padding: 24px 28px 28px; }
    .prompt-line { font-size: 0.78rem; color: var(--green-dim); margin-bottom: 6px; letter-spacing: 0.05em; }
    .prompt-line span { color: var(--green); }
    .sys-title {
      font-family: 'VT323', monospace;
      font-size: 2.8rem;
      line-height: 1;
      color: var(--green);
      text-shadow: 0 0 20px var(--green-glow), 0 0 40px rgba(0,255,65,0.2);
      margin-bottom: 4px;
      letter-spacing: 0.08em;
    }
    .sys-sub { font-size: 0.72rem; color: var(--green-dim); margin-bottom: 20px; letter-spacing: 0.12em; }
    .session-id { font-size: 0.72rem; color: var(--amber); margin-bottom: 22px; letter-spacing: 0.1em; word-break: break-all; }
    .divider { border: none; border-top: 1px solid var(--border); margin: 20px 0; }
    .btn-row { display: grid; grid-template-columns: 1fr; gap: 12px; margin-bottom: 20px; }
    .btn {
      position: relative;
      background: transparent;
      border: 1px solid var(--green-dim);
      color: var(--green);
      font-family: 'Share Tech Mono', monospace;
      font-size: 0.82rem;
      letter-spacing: 0.1em;
      padding: 14px 10px;
      cursor: pointer;
      transition: all 0.15s;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
      text-transform: uppercase;
      overflow: hidden;
    }
    .btn::before {
      content: '';
      position: absolute;
      inset: 0;
      background: var(--green);
      opacity: 0;
      transition: opacity 0.15s;
    }
    .btn:hover { border-color: var(--green); color: var(--bg); box-shadow: 0 0 20px var(--green-glow); }
    .btn:hover::before { opacity: 1; }
    .btn:active { transform: scale(0.98); }
    .btn-icon, .btn-label { position: relative; z-index: 1; }
    .btn-icon { font-family: 'VT323', monospace; font-size: 2rem; line-height: 1; }
    .btn-execute { padding: 16px; border-color: var(--green); background: rgba(0,255,65,0.05); font-size: 0.9rem; }
    .btn-execute::after { content: ' >_'; opacity: 0.6; }
    .status-log {
      background: rgba(0,0,0,0.4);
      border: 1px solid var(--border);
      padding: 10px 14px;
      font-size: 0.7rem;
      color: var(--green-dim);
      max-height: 110px;
      overflow-y: auto;
      letter-spacing: 0.04em;
      line-height: 1.8;
    }
    .status-log::-webkit-scrollbar { width: 4px; }
    .status-log::-webkit-scrollbar-thumb { background: var(--green-dim); }
    .log-line { display: block; }
    .log-line.ok { color: var(--green); }
    .log-line.err { color: var(--red); }
    .log-line.warn { color: var(--amber); }
    .preview-area { display: none; margin-bottom: 20px; border: 1px solid var(--border); padding: 10px; text-align: center; }
    .preview-area.show { display: block; }
    .preview-area img {
      max-width: 100%;
      max-height: 180px;
      filter: grayscale(30%) sepia(20%) hue-rotate(60deg) saturate(150%);
      border: 1px solid var(--green-dim);
    }
    .preview-area .fname { font-size: 0.7rem; color: var(--green-dim); margin-top: 6px; word-break: break-all; }
    .modal-overlay {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.85);
      z-index: 200;
      align-items: center;
      justify-content: center;
    }
    .modal-overlay.show { display: flex; }
    .modal {
      width: min(500px, 95vw);
      border: 1px solid var(--green-dim);
      background: var(--surface);
      box-shadow: 0 0 40px var(--green-glow);
    }
    .modal-header {
      background: #001a00;
      border-bottom: 1px solid var(--green-dim);
      padding: 8px 16px;
      font-family: 'VT323', monospace;
      font-size: 1rem;
      letter-spacing: 0.1em;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .modal-close { background: none; border: none; color: var(--red); font-size: 1.1rem; cursor: pointer; }
    .modal-body { padding: 20px; }
    video, canvas.snap-canvas {
      width: 100%;
      border: 1px solid var(--green-dim);
      display: block;
      filter: grayscale(20%) sepia(10%) hue-rotate(60deg) saturate(130%);
    }
    canvas.snap-canvas { display: none; }
    .modal-btns { display: flex; gap: 10px; margin-top: 14px; }
    .modal-btns .btn { flex: 1; flex-direction: row; padding: 12px; }
  </style>
</head>
<body>
<canvas id="rain"></canvas>
<div class="terminal">
  <div class="term-header">
    <div class="term-dots">
      <div class="dot r"></div><div class="dot a"></div><div class="dot g"></div>
    </div>
    <div class="term-title">root@localhost:~/upload - bash</div>
  </div>
  <div class="term-body">
    <div class="prompt-line"><span>root@h4x</span>:~$ ./init_upload --secure --v2</div>
    <div class="sys-title">SYS::UPLOAD</div>
    <div class="sys-sub">// SECURE FILE INJECTION TERMINAL v2.4.1 //</div>
    <div class="session-id">SESSION::__SESSION_ID__</div>
    <hr class="divider"/>
    <div class="preview-area" id="previewArea">
      <img id="previewImg" src="" alt="preview"/>
      <div class="fname" id="previewName"></div>
    </div>
    <div class="btn-row">
      <button class="btn" id="cameraBtn">
        <span class="btn-icon">[◉]</span>
        <span class="btn-label">Take Photo</span>
      </button>
      <button class="btn btn-execute" id="executeBtn">EXECUTE TRANSMISSION</button>
    </div>
    <div class="status-log" id="statusLog">
      <span class="log-line">[ SYS ] Terminal initialized...</span>
      <span class="log-line ok">[ OK  ] Secure channel established</span>
      <span class="log-line">[ SYS ] Awaiting file input...</span>
    </div>
  </div>
</div>

<div class="modal-overlay" id="cameraModal">
  <div class="modal">
    <div class="modal-header">
      <span>// OPTICAL CAPTURE MODULE //</span>
      <button class="modal-close" id="closeModal">[X]</button>
    </div>
    <div class="modal-body">
      <video id="videoEl" autoplay playsinline></video>
      <canvas class="snap-canvas" id="snapCanvas"></canvas>
      <div class="modal-btns">
        <button class="btn" id="snapBtn"><span class="btn-icon">[◎]</span><span class="btn-label">Capture</span></button>
        <button class="btn" id="retakeBtn" style="display:none"><span class="btn-icon">[↺]</span><span class="btn-label">Retake</span></button>
        <button class="btn" id="usePhotoBtn" style="display:none"><span class="btn-icon">[✔]</span><span class="btn-label">Use Photo</span></button>
      </div>
    </div>
  </div>
</div>
<input type="file" id="cameraInput" accept="image/*" capture="environment"/>

<script>
  (function(){
    const canvas = document.getElementById('rain');
    const ctx = canvas.getContext('2d');
    let cols, drops;
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@#$%^&*<>[]{}|/\\';
    function init(){
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      cols = Math.floor(canvas.width / 16);
      drops = Array(cols).fill(1);
    }
    function draw(){
      ctx.fillStyle = 'rgba(2,10,2,0.05)';
      ctx.fillRect(0,0,canvas.width,canvas.height);
      ctx.fillStyle = '#00ff41';
      ctx.font = '14px Share Tech Mono';
      drops.forEach((y,i)=>{
        const ch = chars[Math.floor(Math.random()*chars.length)];
        ctx.fillText(ch, i*16, y*16);
        if(y*16 > canvas.height && Math.random() > 0.975) drops[i] = 0;
        drops[i]++;
      });
    }
    init();
    window.addEventListener('resize', init);
    setInterval(draw, 50);
  })();

  const statusLog = document.getElementById('statusLog');
  function addLog(msg, cls=''){
    const el = document.createElement('span');
    el.className = 'log-line' + (cls ? ' ' + cls : '');
    el.textContent = msg;
    statusLog.appendChild(el);
    statusLog.scrollTop = statusLog.scrollHeight;
  }

  let selectedFile = null;
  const previewArea = document.getElementById('previewArea');
  const previewImg = document.getElementById('previewImg');
  const previewName = document.getElementById('previewName');
  const executeBtn = document.getElementById('executeBtn');
  executeBtn.disabled = true;
  executeBtn.style.opacity = '0.55';
  executeBtn.style.cursor = 'not-allowed';

  async function handleFile(file, autoSend=false){
    selectedFile = file;
    addLog(`[ IN  ] File queued: ${file.name} (${(file.size/1024).toFixed(1)} KB)`);
    addLog(`[ SYS ] Type: ${file.type || 'unknown'}`, 'ok');
    executeBtn.disabled = false;
    executeBtn.style.opacity = '1';
    executeBtn.style.cursor = 'pointer';
    previewName.textContent = '> ' + file.name;
    if(file.type.startsWith('image/')){
      const reader = new FileReader();
      reader.onload = e => {
        previewImg.src = e.target.result;
        previewArea.classList.add('show');
      };
      reader.readAsDataURL(file);
    } else {
      previewImg.src = '';
      previewArea.classList.remove('show');
    }
    if (autoSend) {
      await transmit();
    }
  }

  async function transmit(){
    if(!selectedFile){
      addLog('[ SYS ] No photo selected yet. Opening camera...', 'warn');
      cameraInput.click();
      return;
    }
    executeBtn.disabled = true;
    executeBtn.style.opacity = '0.55';
    executeBtn.style.cursor = 'not-allowed';
    addLog('[ SYS ] Initiating secure transmission...', 'warn');
    const formData = new FormData();
    formData.append('photo', selectedFile, selectedFile.name || 'capture.jpg');
    try {
      const res = await fetch('/session/__SESSION_ID__/upload', {
        method: 'POST',
        body: formData
      });
      const text = await res.text();
      if (res.ok) {
        addLog('[ OK  ] Transmission complete. Proof uploaded.', 'ok');
        addLog(`[ SYS ] ${text}`, 'ok');
        selectedFile = null;
        previewArea.classList.remove('show');
        previewName.textContent = '';
      } else {
        addLog(`[ ERR ] Upload failed (${res.status})`, 'err');
        addLog(`[ ERR ] ${text}`, 'err');
        executeBtn.disabled = false;
        executeBtn.style.opacity = '1';
        executeBtn.style.cursor = 'pointer';
      }
    } catch (err) {
      addLog(`[ ERR ] Network error: ${err.message}`, 'err');
      executeBtn.disabled = false;
      executeBtn.style.opacity = '1';
      executeBtn.style.cursor = 'pointer';
    }
  }

  executeBtn.addEventListener('click', transmit);

  const modal = document.getElementById('cameraModal');
  const videoEl = document.getElementById('videoEl');
  const snapCanvas = document.getElementById('snapCanvas');
  const snapBtn = document.getElementById('snapBtn');
  const retakeBtn = document.getElementById('retakeBtn');
  const usePhotoBtn = document.getElementById('usePhotoBtn');
  const cameraInput = document.getElementById('cameraInput');
  let stream = null;

  function stopStream(){
    if(stream){ stream.getTracks().forEach(t=>t.stop()); stream = null; }
  }

  document.getElementById('cameraBtn').addEventListener('click', async ()=>{
    if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia){
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
        videoEl.srcObject = stream;
        videoEl.style.display = '';
        snapCanvas.style.display = 'none';
        snapBtn.style.display = '';
        retakeBtn.style.display = 'none';
        usePhotoBtn.style.display = 'none';
        modal.classList.add('show');
        addLog('[ CAM ] Optical module online', 'ok');
      } catch(err){
        addLog('[ ERR ] Camera access denied: ' + err.message, 'err');
        cameraInput.click();
      }
    } else {
      cameraInput.click();
    }
  });

  cameraInput.addEventListener('change', async e=>{
    if(e.target.files && e.target.files[0]) await handleFile(e.target.files[0], true);
  });

  document.getElementById('closeModal').addEventListener('click', ()=>{
    stopStream();
    modal.classList.remove('show');
    addLog('[ CAM ] Optical module offline', 'warn');
  });

  snapBtn.addEventListener('click', ()=>{
    snapCanvas.width = videoEl.videoWidth;
    snapCanvas.height = videoEl.videoHeight;
    snapCanvas.getContext('2d').drawImage(videoEl, 0, 0);
    videoEl.style.display = 'none';
    snapCanvas.style.display = '';
    snapBtn.style.display = 'none';
    retakeBtn.style.display = '';
    usePhotoBtn.style.display = '';
    addLog('[ CAM ] Frame captured', 'ok');
  });

  retakeBtn.addEventListener('click', ()=>{
    videoEl.style.display = '';
    snapCanvas.style.display = 'none';
    snapBtn.style.display = '';
    retakeBtn.style.display = 'none';
    usePhotoBtn.style.display = 'none';
  });

  usePhotoBtn.addEventListener('click', ()=>{
    snapCanvas.toBlob(blob=>{
      if (!blob) {
        addLog('[ ERR ] Capture failed. Try again.', 'err');
        return;
      }
      const file = new File([blob], `capture_${Date.now()}.png`, { type: 'image/png' });
      handleFile(file, true);
    }, 'image/png');
    stopStream();
    modal.classList.remove('show');
    addLog('[ CAM ] Image committed to buffer', 'ok');
  });
</script>
</body>
</html>"""
    return page.replace("__SESSION_ID__", session_id)


def _build_mobile_upload_page(session_id: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Garden Check-in</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: linear-gradient(180deg, #111420 0%, #171b2b 56%, #132022 100%);
      color: #e7ebff;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 16px;
    }}
    .card {{
      width: min(430px, 96vw);
      background: #23283b;
      border: 1px solid #50597a;
      border-radius: 18px;
      box-shadow: 0 12px 34px rgba(12, 14, 24, 0.55);
      padding: 18px;
    }}
    .title {{ font-size: 1.25rem; font-weight: 700; margin: 4px 0 4px; }}
    .sub {{ color: #bec7ea; font-size: 0.94rem; line-height: 1.4; margin-bottom: 14px; }}
    .tag {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      background: #3b2f59;
      color: #d6c5ff;
      font-size: 0.75rem;
      font-weight: 700;
      margin-bottom: 10px;
    }}
    .btn {{
      width: 100%;
      border: 0;
      border-radius: 12px;
      padding: 14px;
      font-weight: 700;
      font-size: 1rem;
      cursor: pointer;
      margin-top: 10px;
    }}
    .primary {{ background: #8f73e6; color: #fff; }}
    .primary:disabled {{ background: #9ca3af; cursor: not-allowed; }}
    .ghost {{ background: #333a57; color: #d7ddfb; }}
    .preview {{
      margin-top: 14px;
      display: none;
      border: 1px solid #4f5879;
      border-radius: 12px;
      overflow: hidden;
      background: #1f2436;
    }}
    .preview img {{ width: 100%; display: block; }}
    .status {{
      margin-top: 12px;
      font-size: 0.9rem;
      background: #1d2335;
      border: 1px solid #4a5474;
      border-radius: 10px;
      padding: 10px;
      color: #dde3ff;
    }}
    .badge-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-top: 10px;
      margin-bottom: 8px;
    }}
    .badge-pill {{
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 0.74rem;
      font-weight: 700;
      letter-spacing: 0.03em;
      border: 1px solid #546084;
      background: #272f45;
      color: #dce6ff;
      white-space: nowrap;
    }}
    .badge-live {{
      background: #1f3b2f;
      border-color: #4ec78f;
      color: #aff6d0;
    }}
    .badge-fallback {{
      background: #3f3720;
      border-color: #d1b06a;
      color: #ffe8b0;
    }}
    .badge-retry {{
      background: #3f2328;
      border-color: #db7f90;
      color: #ffd0d8;
    }}
    .badge-mini {{
      font-size: 0.72rem;
      color: #b9c4ec;
    }}
  </style>
</head>
<body>
  <div class="card">
    <div class="tag">Garden check-in active</div>
    <div class="title">Take a quick outdoor photo</div>
    <div class="sub">Include grass/plants and outdoor context. Once selected, upload happens automatically.</div>

    <div class="badge-row">
      <div id="chainBadge" class="badge-pill badge-fallback">CHAIN_SIMULATED_IMMUTABLE</div>
      <div id="sessionMeta" class="badge-mini">session: {session_id[:8]}</div>
    </div>

    <input id="cameraInput" type="file" accept="image/*" capture="environment" style="display:none" />
    <button id="pickBtn" class="btn ghost">Open Camera</button>
    <button id="sendBtn" class="btn primary" disabled>Upload Check-in</button>

    <div id="preview" class="preview"><img id="previewImg" alt="Preview"/></div>
    <div id="status" class="status">Ready for capture.</div>
  </div>

  <script>
    let selectedFile = null;
    const cameraInput = document.getElementById('cameraInput');
    const pickBtn = document.getElementById('pickBtn');
    const sendBtn = document.getElementById('sendBtn');
    const preview = document.getElementById('preview');
    const previewImg = document.getElementById('previewImg');
    const statusEl = document.getElementById('status');
    const chainBadgeEl = document.getElementById('chainBadge');
    const sessionMetaEl = document.getElementById('sessionMeta');

    function setStatus(msg) {{
      statusEl.textContent = msg;
    }}

    async function transmit() {{
      if (!selectedFile) {{
        cameraInput.click();
        return;
      }}
      sendBtn.disabled = true;
      setStatus('Uploading...');
      const fd = new FormData();
      fd.append('photo', selectedFile, selectedFile.name || `capture_${{Date.now()}}.jpg`);
      try {{
        const res = await fetch('/session/{session_id}/upload', {{ method: 'POST', body: fd }});
        const text = await res.text();
        if (res.ok) {{
          setStatus('Uploaded. Return to laptop to continue.');
          sendBtn.textContent = 'Uploaded';
          preview.style.display = 'none';
          selectedFile = null;
        }} else {{
          setStatus(`Upload failed (${{res.status}}): ${{text}}`);
          sendBtn.disabled = false;
        }}
      }} catch (err) {{
        setStatus('Network error. Check Wi-Fi and retry.');
        sendBtn.disabled = false;
      }}
    }}

    function setBadgeStyle(value) {{
      chainBadgeEl.classList.remove('badge-live', 'badge-fallback', 'badge-retry');
      if (value === 'CHAIN_LIVE') {{
        chainBadgeEl.classList.add('badge-live');
      }} else if (value === 'CHAIN_RETRYING') {{
        chainBadgeEl.classList.add('badge-retry');
      }} else {{
        chainBadgeEl.classList.add('badge-fallback');
      }}
    }}

    async function pollStatus() {{
      try {{
        const res = await fetch('/session/{session_id}/status');
        if (!res.ok) return;
        const data = await res.json();
        const badge = data.chain_badge || 'CHAIN_SIMULATED_IMMUTABLE';
        chainBadgeEl.textContent = badge;
        setBadgeStyle(badge);
        sessionMetaEl.textContent = `level: ${{data.flower_level || 1}} | value: $${{(data.value_score || 0).toFixed(2)}}`;
      }} catch (_) {{
        // keep previous values; network hiccups are expected on mobile
      }}
    }}

    pickBtn.addEventListener('click', () => cameraInput.click());
    sendBtn.addEventListener('click', transmit);
    cameraInput.addEventListener('change', async (e) => {{
      const f = e.target.files && e.target.files[0];
      if (!f) return;
      selectedFile = f;
      sendBtn.disabled = false;
      sendBtn.textContent = 'Upload Proof';
      const reader = new FileReader();
      reader.onload = (ev) => {{
        previewImg.src = ev.target.result;
        preview.style.display = 'block';
      }};
      reader.readAsDataURL(f);
      setStatus(`Selected: ${{f.name}}. Uploading now...`);
      await transmit();
    }});

    setBadgeStyle(chainBadgeEl.textContent);
    pollStatus();
    setInterval(pollStatus, 1200);
  </script>
</body>
</html>
"""


def create_upload_app(upload_state: UploadState) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def home() -> str:
        with upload_state.lock:
            active_session_id = upload_state.active_session_id
        if active_session_id:
            return f"""
<!doctype html>
<html>
  <head><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
  <body style="font-family:Arial;padding:20px;">
    <h2>Recovery Session Active</h2>
    <p>Continue via session page:</p>
    <a href="/session/{active_session_id}">Open active recovery upload</a>
    <p style="margin-top:20px;"><a href="/my-flower">View My Flower Token</a></p>
  </body>
</html>
            """
        return """
<!doctype html>
<html>
  <head><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
  <body style="font-family:Arial;padding:20px;">
    <h2>Touch Grass Upload</h2>
    <p>Scan the QR from laptop after entering Recovery Required mode.</p>
    <p><a href="/my-flower">Open flower wallet view</a></p>
  </body>
</html>
        """

    @app.get("/my-flower")
    def my_flower() -> str:
        with upload_state.lock:
            token_id = upload_state.latest_token_id
            proof_url = upload_state.latest_proof_url
            level = upload_state.latest_flower_level
            value = upload_state.latest_value
            block_hash = upload_state.latest_block_hash
            chain_badge = upload_state.latest_chain_badge
        proof_link = (
            f'<a href="{proof_url}" target="_blank" rel="noreferrer">Open Public Proof</a>'
            if proof_url
            else "Proof link pending (set NFT_STORAGE_API_KEY for IPFS link)."
        )
        token_text = token_id if token_id else "Pending first verified session"
        return f"""
<!doctype html>
<html>
  <head><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
  <body style="font-family:Arial;padding:20px;background:linear-gradient(180deg,#111420,#171b2b,#132022);color:#e7ebff;">
    <h2>My Flower Wallet View</h2>
    <p>Token: <b>{token_text}</b></p>
    <p>Flower Level: <b>{level}</b></p>
    <p>Value Score: <b>${value:.2f}</b></p>
    <p>Chain Badge: <b>{chain_badge}</b></p>
    <p>Local Chain Block: <b>{block_hash or "Pending first verification"}</b></p>
    <p>{proof_link}</p>
    <p><a href="/" style="color:#bda9ff;">Back to upload home</a></p>
  </body>
</html>
        """

    @app.get("/session/<session_id>")
    def session_page(session_id: str) -> tuple[str, int]:
        with upload_state.lock:
            is_active = session_id == upload_state.active_session_id
        if not is_active:
            return "Session not active. Re-scan a fresh QR from laptop.", 404
        return (_build_mobile_upload_page(session_id), 200)

    @app.get("/session/<session_id>/status")
    def session_status(session_id: str):
        with upload_state.lock:
            is_active = session_id == upload_state.active_session_id
            payload = {
                "active": is_active,
                "chain_badge": upload_state.latest_chain_badge,
                "flower_level": upload_state.latest_flower_level,
                "value_score": float(upload_state.latest_value),
                "token_id": upload_state.latest_token_id,
                "proof_url": upload_state.latest_proof_url,
                "block_hash": upload_state.latest_block_hash,
            }
        return jsonify(payload), 200

    @app.post("/session/<session_id>/upload")
    def upload(session_id: str) -> tuple[str, int]:
        with upload_state.lock:
            is_active = session_id == upload_state.active_session_id
        if not is_active:
            return "Session expired. Start a new recovery session on laptop.", 400
        photo = request.files.get("photo")
        if photo is None:
            return "Missing photo field.", 400
        raw = photo.read()
        buf = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return "Invalid image.", 400

        with upload_state.lock:
            upload_state.latest_frame = frame
            upload_state.upload_count += 1
            upload_state.latest_status = f"Upload received for session {session_id}"
            upload_state.latest_session_id = session_id
            upload_state.latest_received_at = time.time()
            upload_state.session_upload_counts[session_id] = (
                upload_state.session_upload_counts.get(session_id, 0) + 1
            )
        return (
            "Upload received. Return to laptop app for grass+outside verification verdict.",
            200,
        )

    return app


def start_upload_server(upload_state: UploadState, host: str, port: int) -> None:
    app = create_upload_app(upload_state)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    def run() -> None:
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()


def draw_debug_overlay(
    frame_bgr: np.ndarray,
    signals: TrackingSignals,
    app_state: RecoveryAppState,
    chain_badge: ChainBadge,
    live_metrics: LiveMetrics,
    session: RecoverySession | None,
    prompt_visible: bool,
    screen_time_seconds: float,
    focus_streak_seconds: float,
    flower_assets: list[np.ndarray],
    mint_flash_until: float,
    latest_proof_url: str,
    qr_tile: np.ndarray | None = None,
    status_message: str | None = None,
    demo_step_text: str = "",
) -> np.ndarray:
    now = time.time()
    # Keep camera window minimal: only tiny status strip.
    canvas = frame_bgr.copy()
    h_total, w_total = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w_total, 26), (18, 20, 28), -1)
    cv2.putText(
        canvas,
        f"{app_state.value} | look {signals.looking_score:.2f} | eyes {'YES' if signals.eyes_detected else 'NO'}",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (232, 236, 255),
        1,
        cv2.LINE_AA,
    )

    if prompt_visible:
        cv2.rectangle(canvas, (8, h_total - 46), (w_total - 8, h_total - 8), (35, 41, 61), -1)
        cv2.rectangle(canvas, (8, h_total - 46), (w_total - 8, h_total - 8), (132, 146, 188), 1)
        cv2.putText(canvas, "Press Y/Enter/Space (or R force) for outdoor check-in", (16, h_total - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (228, 234, 255), 1, cv2.LINE_AA)

    # Only show the big NFT pop once minted.
    if mint_flash_until > now:
        alpha = min(1.0, max(0.25, (mint_flash_until - now) / 6.0))
        flash = canvas.copy()
        cx = max(18, (w_total // 2) - 140)
        cy = max(30, (h_total // 2) - 52)
        cv2.rectangle(flash, (cx, cy), (cx + 280, cy + 104), (20, 76, 42), -1)
        cv2.rectangle(flash, (cx, cy), (cx + 280, cy + 104), (137, 235, 170), 2)
        cv2.putText(flash, "NFT UPGRADED", (cx + 34, cy + 40), cv2.FONT_HERSHEY_DUPLEX, 0.75, (242, 255, 245), 2, cv2.LINE_AA)
        cv2.putText(flash, f"${max(0.0, live_metrics.score_value*10.0):.2f}", (cx + 98, cy + 78), cv2.FONT_HERSHEY_DUPLEX, 0.82, (232, 245, 238), 2, cv2.LINE_AA)
        cv2.addWeighted(flash, alpha, canvas, 1.0 - alpha, 0, canvas)

    return canvas


def _demo_step_text(app_state: RecoveryAppState) -> str:
    mapping = {
        RecoveryAppState.ONLINE: "Step 1: Calm baseline. Plant is comfy.",
        RecoveryAppState.WARNING: "Step 2: Gentle reminder to take a break soon.",
        RecoveryAppState.DECAY: "Step 3: Plant mood dropping a bit.",
        RecoveryAppState.RECOVERY_REQUIRED: "Step 4: Press Y to begin outdoor check-in.",
        RecoveryAppState.OUTSIDE_MODE: "Step 5: Scan QR + share an outdoor photo.",
        RecoveryAppState.PROOF_VERIFIED: "Step 6: Great shot! Updating flower value...",
        RecoveryAppState.MINTED: "Step 7: Flower upgraded and value increased.",
    }
    return mapping.get(app_state, "")


def _build_nft_metadata(
    wallet_address: str,
    metrics: LiveMetrics,
    session: RecoverySession,
) -> dict:
    return {
        "name": "Grass Token",
        "owner": wallet_address,
        "outdoor_sessions": metrics.outdoor_sessions + 1,
        "streak_days": max(metrics.streak_days, 1),
        "plant_health": int(min(100, 30 + 6 * (metrics.outdoor_sessions + 1))),
        "screen_balance": round(metrics.score_value + 0.75, 2),
        "latest_session": {
            "session_id": session.session_id,
            "vegetation_score": round(session.vegetation_score, 3),
            "outdoor_score": round(session.outdoor_score, 3),
        },
    }


def _upload_to_nft_storage(
    api_key: str,
    payload: bytes,
    content_type: str,
) -> str:
    if not api_key.strip():
        return ""
    req = urllib.request.Request(
        "https://api.nft.storage/upload",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": content_type,
        },
    )
    with urllib.request.urlopen(req, timeout=20.0) as resp:
        raw = resp.read().decode("utf-8")
    parsed = json.loads(raw) if raw else {}
    value = parsed.get("value", {}) if isinstance(parsed, dict) else {}
    cid = str(value.get("cid", "")).strip()
    return cid


def _simulate_local_mint_with_proof(
    session: RecoverySession,
    wallet_address: str,
    metadata: dict,
    upload_frame: np.ndarray | None,
    existing_token_id: str,
    nft_storage_api_key: str,
    artifacts_dir: str,
) -> ChainActionResult:
    os.makedirs(artifacts_dir, exist_ok=True)
    image_path = os.path.join(artifacts_dir, f"{session.session_id}_proof.jpg")
    metadata_path = os.path.join(artifacts_dir, f"{session.session_id}_metadata.json")
    proof_url = ""
    error = ""

    if upload_frame is not None:
        cv2.imwrite(image_path, upload_frame)

    image_cid = ""
    metadata_cid = ""
    try:
        if nft_storage_api_key.strip() and os.path.isfile(image_path):
            with open(image_path, "rb") as fh:
                image_cid = _upload_to_nft_storage(
                    api_key=nft_storage_api_key,
                    payload=fh.read(),
                    content_type="image/jpeg",
                )
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        error = f"image_upload_failed:{exc}"

    metadata_out = dict(metadata)
    metadata_out["proof_mode"] = "local_emergency_chain_ready"
    metadata_out["session_id"] = session.session_id
    metadata_out["proof_image_local"] = image_path if os.path.isfile(image_path) else ""
    if image_cid:
        metadata_out["image"] = f"ipfs://{image_cid}"
        metadata_out["proof_image_public"] = f"https://nftstorage.link/ipfs/{image_cid}"

    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata_out, fh, indent=2, sort_keys=True)

    try:
        if nft_storage_api_key.strip():
            with open(metadata_path, "rb") as fh:
                metadata_cid = _upload_to_nft_storage(
                    api_key=nft_storage_api_key,
                    payload=fh.read(),
                    content_type="application/json",
                )
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        if error:
            error = f"{error};metadata_upload_failed:{exc}"
        else:
            error = f"metadata_upload_failed:{exc}"

    if metadata_cid:
        proof_url = f"https://nftstorage.link/ipfs/{metadata_cid}"

    token_id = existing_token_id or str(int(time.time()))
    tx_seed = f"{session.session_id}:{wallet_address}:{token_id}:{proof_url}:{time.time_ns()}"
    tx_hash = "0x" + hashlib.sha256(tx_seed.encode("utf-8")).hexdigest()
    return ChainActionResult(
        badge=ChainBadge.CHAIN_FALLBACK_LOCAL_LEDGER,
        token_id=token_id,
        tx_hash=tx_hash,
        error=error,
        proof_url=proof_url,
        metadata_path=metadata_path,
    )


def _overlay_stage_from_app_state(app_state: RecoveryAppState) -> str:
    mapping = {
        RecoveryAppState.ONLINE: "normal",
        RecoveryAppState.WARNING: "warning",
        RecoveryAppState.DECAY: "decay",
        RecoveryAppState.RECOVERY_REQUIRED: "decay",
        RecoveryAppState.OUTSIDE_MODE: "recovery",
        RecoveryAppState.PROOF_VERIFIED: "recovery",
        RecoveryAppState.MINTED: "recovery",
    }
    return mapping.get(app_state, "normal")


def _write_overlay_stage(stage_file: str, app_state: RecoveryAppState) -> None:
    try:
        with open(stage_file, "w", encoding="utf-8") as fh:
            fh.write(_overlay_stage_from_app_state(app_state))
    except OSError as exc:
        print(f"[TouchGrass] Could not write stage file '{stage_file}': {exc}")


def _write_overlay_status(
    status_file: str,
    payload: dict,
) -> None:
    try:
        os.makedirs(os.path.dirname(status_file), exist_ok=True)
        with open(status_file, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    except OSError as exc:
        print(f"[TouchGrass] Could not write overlay status '{status_file}': {exc}")


def _display_chain_badge(raw_badge: str) -> str:
    if raw_badge == ChainBadge.CHAIN_LIVE.value:
        return "CHAIN_LIVE"
    if raw_badge == ChainBadge.CHAIN_RETRYING.value:
        return "CHAIN_RETRYING"
    return "CHAIN_SIMULATED_IMMUTABLE"


def _run_chain_smoke_test(
    chain_bridge: ThirdwebBridge,
    wallet_address: str,
    existing_token_id: str,
) -> int:
    metadata = {
        "name": "TouchGrass Chain Smoke Test",
        "owner": wallet_address,
        "smoke_test": True,
        "timestamp": int(time.time()),
        "notes": "Dry run from camera_tracking.py --chain-smoke-test",
    }
    result = chain_bridge.mint_or_update(
        wallet_address=wallet_address,
        metadata=metadata,
        existing_token_id=existing_token_id,
    )
    print(
        json.dumps(
            {
                "badge": result.badge.value,
                "token_id": result.token_id,
                "tx_hash": result.tx_hash,
                "error": result.error,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.badge == ChainBadge.CHAIN_LIVE else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen recovery + outdoor proof tracker.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--training-folder", type=str, default="photos")
    parser.add_argument("--upload-host-ip", type=str, default="")
    parser.add_argument("--upload-port", type=int, default=8088)
    parser.add_argument("--camera-open-timeout", type=float, default=8.0)
    parser.add_argument("--looking-threshold", type=float, default=0.62)
    parser.add_argument("--vegetation-threshold", type=float, default=0.24)
    parser.add_argument("--outdoor-threshold", type=float, default=0.35)
    parser.add_argument("--warning-hold-seconds", type=float, default=3.0)
    parser.add_argument("--look-hold-seconds", type=float, default=6.0)
    parser.add_argument("--decay-hold-seconds", type=float, default=0.0)
    parser.add_argument("--decay-stage-seconds", type=float, default=2.0)
    parser.add_argument("--min-absence-seconds", type=float, default=2.5)
    parser.add_argument("--session-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--proof-result-seconds", type=float, default=8.0)
    parser.add_argument("--demo-mode", action="store_true")
    parser.add_argument("--mint-enabled", action="store_true")
    parser.add_argument("--wallet-address", type=str, default="demo_wallet")
    parser.add_argument("--thirdweb-mint-url", type=str, default=os.getenv("THIRDWEB_MINT_URL", ""))
    parser.add_argument("--thirdweb-update-url", type=str, default=os.getenv("THIRDWEB_UPDATE_URL", ""))
    parser.add_argument("--thirdweb-api-key", type=str, default=os.getenv("THIRDWEB_API_KEY", ""))
    parser.add_argument("--emergency-mode", action="store_true", default=os.getenv("EMERGENCY_MODE", "0") == "1")
    parser.add_argument("--nft-storage-api-key", type=str, default=os.getenv("NFT_STORAGE_API_KEY", ""))
    parser.add_argument("--emergency-artifacts-dir", type=str, default=os.path.join("data", "proof_artifacts"))
    parser.add_argument("--flowers-folder", type=str, default="flowers")
    parser.add_argument("--overlay-stage-file", type=str, default="stage.txt")
    parser.add_argument("--overlay-status-file", type=str, default=os.path.join("data", "overlay_status.json"))
    parser.add_argument("--overlay-qr-file", type=str, default=os.path.join("data", "current_qr.png"))
    parser.add_argument("--auto-recovery-start", action="store_true")
    parser.add_argument("--chain-smoke-test", action="store_true")
    parser.add_argument("--chain-smoke-token-id", type=str, default="")
    args = parser.parse_args()

    if args.demo_mode:
        args.warning_hold_seconds = 2.0
        args.look_hold_seconds = 6.0
        args.decay_stage_seconds = 1.6
        args.min_absence_seconds = 1.8
        args.session_timeout_seconds = 90.0
        args.auto_recovery_start = True

    chain_bridge = ThirdwebBridge(
        enabled=args.mint_enabled,
        mint_url=args.thirdweb_mint_url,
        update_url=args.thirdweb_update_url,
        api_key=args.thirdweb_api_key,
    )
    if args.chain_smoke_test:
        if not args.mint_enabled:
            print("[TouchGrass] --chain-smoke-test requires --mint-enabled.")
            return
        exit_code = _run_chain_smoke_test(
            chain_bridge=chain_bridge,
            wallet_address=args.wallet_address,
            existing_token_id=args.chain_smoke_token_id.strip(),
        )
        if exit_code != 0:
            print("[TouchGrass] Chain smoke test failed (non-live badge). Check endpoint/key.")
        return

    decay_hold_seconds = args.decay_hold_seconds if args.decay_hold_seconds > 0 else args.look_hold_seconds

    tracker = CameraTracker(
        looking_threshold=args.looking_threshold,
        vegetation_threshold=args.vegetation_threshold,
        outdoor_threshold=args.outdoor_threshold,
    )
    train_stats = tracker.bootstrap_training_from_folder(args.training_folder)
    flower_assets = _load_flower_assets(args.flowers_folder)
    upload_state = UploadState()
    local_ip = args.upload_host_ip.strip() or get_local_ip()
    upload_root_url = f"http://{local_ip}:{args.upload_port}/"
    os.makedirs(os.path.dirname(args.overlay_qr_file), exist_ok=True)
    qr_tile = make_qr_tile(upload_root_url, size_px=240)
    cv2.imwrite(args.overlay_qr_file, qr_tile)
    start_upload_server(upload_state, host="0.0.0.0", port=args.upload_port)
    print(f"[TouchGrass] Upload URL: {upload_root_url}")

    ledger = RecoveryLedger(os.path.join("data", "recovery_ledger.json"))
    local_chain = LocalImmutableChain(os.path.join("data", "simulated_chain.jsonl"))
    live_metrics = ledger.metrics()
    with upload_state.lock:
        upload_state.latest_flower_level = max(1, live_metrics.outdoor_sessions + 1)
        upload_state.latest_value = live_metrics.score_value
        upload_state.latest_token_id = live_metrics.token_id
        upload_state.latest_chain_badge = _display_chain_badge(ChainBadge.CHAIN_FALLBACK_LOCAL_LEDGER.value)

    cam = open_camera_with_retry(args.camera_index, args.camera_open_timeout)
    if cam is None:
        raise RuntimeError("Could not open laptop camera.")
    window_name = "Hackathon Camera Tracking (q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if args.demo_mode:
        cv2.resizeWindow(window_name, 380, 240)
        cv2.moveWindow(window_name, 18, 360)

    app_state = RecoveryAppState.ONLINE
    chain_badge = ChainBadge.CHAIN_FALLBACK_LOCAL_LEDGER
    state_started_at = time.time()
    look_started_at: float | None = None
    absent_started_at: float | None = None
    prompt_visible = False
    current_session: RecoverySession | None = None
    last_processed_upload_count = 0
    result_state_until = 0.0
    mint_flash_until = 0.0
    latest_proof_url = ""
    latest_block_hash = ""
    last_mint_at = 0.0
    screen_time_seconds = 0.0
    focus_streak_seconds = 0.0
    last_frame_ts = time.time()
    last_overlay_status_write = 0.0

    status_message = (
        "Preflight ready. "
        f"train seen={train_stats['files_seen']} loaded={train_stats['files_loaded']} "
        f"veg+={train_stats['veg_pos']} veg-={train_stats['veg_neg']} "
        f"out+={train_stats['out_pos']} out-={train_stats['out_neg']}."
    )
    status_message_until = time.time() + 8.0
    _write_overlay_stage(args.overlay_stage_file, app_state)
    _write_overlay_status(
        args.overlay_status_file,
        {
            "ts": time.time(),
            "state": app_state.value,
            "chain_badge": _display_chain_badge(chain_badge.value),
            "face_detected": False,
            "eyes_detected": False,
            "looking_score": 0.0,
            "screen_time_seconds": 0.0,
            "focus_streak_seconds": 0.0,
            "status_message": status_message,
            "demo_step": _demo_step_text(app_state) if args.demo_mode else "",
            "flower_level": max(1, live_metrics.outdoor_sessions + 1),
            "health_score": int(np.clip(56 + (7 * live_metrics.outdoor_sessions) - (8 * live_metrics.decay_penalties), 0, 100)),
            "value_score": round(live_metrics.score_value, 2),
            "sessions": live_metrics.outdoor_sessions,
            "streak_days": live_metrics.streak_days,
            "decay_penalties": live_metrics.decay_penalties,
            "token_id": live_metrics.token_id,
            "tx_hash": live_metrics.latest_tx_hash,
            "proof_url": "",
            "local_block_hash": "",
            "last_mint_at": 0.0,
            "mint_flash_until": 0.0,
            "qr_file": args.overlay_qr_file,
        },
    )

    def set_state(new_state: RecoveryAppState) -> None:
        nonlocal app_state, state_started_at, live_metrics
        if new_state == app_state:
            return
        app_state = new_state
        state_started_at = time.time()
        _write_overlay_stage(args.overlay_stage_file, app_state)
        if new_state == RecoveryAppState.DECAY:
            live_metrics = ledger.mark_decay()

    def start_recovery_session(now: float, signals: TrackingSignals, allow_no_face: bool = False) -> bool:
        nonlocal current_session, qr_tile, prompt_visible, absent_started_at, status_message, status_message_until
        if current_session is not None:
            return False
        if not signals.face_detected and not allow_no_face:
            status_message = "Cannot start recovery: face must be visible at session start."
            status_message_until = now + 3.0
            return False
        session_id = uuid.uuid4().hex[:12]
        current_session = RecoverySession(
            session_id=session_id,
            started_at=now,
            state_started_at=now,
            user_present_at_start=signals.face_detected,
        )
        with upload_state.lock:
            upload_state.active_session_id = session_id
        session_url = f"http://{local_ip}:{args.upload_port}/session/{session_id}"
        qr_tile = make_qr_tile(session_url, size_px=240)
        cv2.imwrite(args.overlay_qr_file, qr_tile)
        set_state(RecoveryAppState.OUTSIDE_MODE)
        prompt_visible = False
        absent_started_at = None
        if allow_no_face and not signals.face_detected:
            status_message = "Recovery auto-started for demo. Scan QR from phone and go outside."
        else:
            status_message = "Recovery session started. Scan QR from phone and go outside."
        status_message_until = now + 8.0
        return True

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                continue
            now = time.time()
            signals = tracker.process_looking_only(frame)
            frame_dt = max(0.0, min(0.25, now - last_frame_ts))
            last_frame_ts = now
            if signals.face_detected and signals.looking_at_screen:
                screen_time_seconds += frame_dt
                focus_streak_seconds += frame_dt
            else:
                focus_streak_seconds = 0.0

            # Main state progression while not currently handling a session.
            if current_session is None and app_state not in (RecoveryAppState.PROOF_VERIFIED, RecoveryAppState.MINTED):
                if signals.looking_at_screen:
                    if look_started_at is None:
                        look_started_at = now
                    look_elapsed = now - look_started_at
                    if look_elapsed < args.warning_hold_seconds:
                        set_state(RecoveryAppState.ONLINE)
                        prompt_visible = False
                    elif look_elapsed < decay_hold_seconds:
                        set_state(RecoveryAppState.WARNING)
                        prompt_visible = False
                    elif look_elapsed < (decay_hold_seconds + args.decay_stage_seconds):
                        set_state(RecoveryAppState.DECAY)
                        prompt_visible = False
                    else:
                        set_state(RecoveryAppState.RECOVERY_REQUIRED)
                        prompt_visible = True
                else:
                    look_started_at = None
                    prompt_visible = False
                    set_state(RecoveryAppState.ONLINE)

            # Session runtime checks for OUTSIDE_MODE.
            if current_session is not None and app_state == RecoveryAppState.OUTSIDE_MODE:
                if now - current_session.started_at > args.session_timeout_seconds:
                    current_session.proof_status = "session_timeout"
                    current_session.completed_at = now
                    with upload_state.lock:
                        upload_state.active_session_id = ""
                    current_session = None
                    set_state(RecoveryAppState.ONLINE)
                    qr_tile = make_qr_tile(upload_root_url, size_px=240)
                    cv2.imwrite(args.overlay_qr_file, qr_tile)
                    status_message = "Session timed out. Recovery reset."
                    status_message_until = now + 5.0
                else:
                    if signals.face_detected:
                        absent_started_at = None
                    else:
                        if absent_started_at is None:
                            absent_started_at = now
                            if current_session.left_screen_at is None:
                                current_session.left_screen_at = now
                        if (
                            current_session.absence_validated_at is None
                            and now - absent_started_at >= args.min_absence_seconds
                        ):
                            current_session.absence_validated_at = now
                            status_message = "Absence validated. Upload outdoor proof from phone now."
                            status_message_until = now + 6.0

            # Process newly uploaded photo once, and only for active session.
            with upload_state.lock:
                upload_count = upload_state.upload_count
                upload_frame = upload_state.latest_frame.copy() if upload_state.latest_frame is not None else None
                upload_session_id = upload_state.latest_session_id

            if upload_count > last_processed_upload_count and upload_frame is not None:
                last_processed_upload_count = upload_count
                if current_session is None or upload_session_id != current_session.session_id:
                    status_message = "Ignored upload from stale/non-active session."
                    status_message_until = now + 3.0
                elif current_session.absence_validated_at is None and not args.demo_mode:
                    status_message = "Upload rejected: leave screen first (absence not yet validated)."
                    status_message_until = now + 4.0
                else:
                    current_session.upload_count += 1
                    upload_signals = tracker.infer_upload(upload_frame)
                    current_session.vegetation_score = upload_signals.vegetation_score
                    current_session.outdoor_score = upload_signals.outdoor_score
                    current_session.grass_outside_score = upload_signals.grass_outside_score
                    demo_verified = (
                        args.demo_mode
                        and (
                            upload_signals.grass_outside_detected
                            or (upload_signals.outdoor_score >= 0.20 and upload_signals.vegetation_score >= 0.16)
                            or upload_signals.outdoor_score >= 0.32
                        )
                    )
                    if upload_signals.grass_outside_detected or demo_verified:
                        current_session.proof_status = "verified"
                        current_session.completed_at = now
                        set_state(RecoveryAppState.PROOF_VERIFIED)
                        metadata = _build_nft_metadata(args.wallet_address, live_metrics, current_session)
                        if args.emergency_mode:
                            chain_result = _simulate_local_mint_with_proof(
                                session=current_session,
                                wallet_address=args.wallet_address,
                                metadata=metadata,
                                upload_frame=upload_frame,
                                existing_token_id=live_metrics.token_id,
                                nft_storage_api_key=args.nft_storage_api_key,
                                artifacts_dir=args.emergency_artifacts_dir,
                            )
                        else:
                            chain_result = chain_bridge.mint_or_update(
                                wallet_address=args.wallet_address,
                                metadata=metadata,
                                existing_token_id=live_metrics.token_id,
                            )
                        if chain_result.badge == ChainBadge.CHAIN_RETRYING:
                            # Reliability fallback for demos: still record behavior + value locally.
                            chain_result = ChainActionResult(
                                badge=ChainBadge.CHAIN_FALLBACK_LOCAL_LEDGER,
                                token_id=live_metrics.token_id,
                                error=chain_result.error,
                            )

                        # Always append immutable local-chain block for demo auditability.
                        chain_index, local_block_hash = local_chain.append_event(
                            {
                                "session_id": current_session.session_id,
                                "wallet": args.wallet_address,
                                "token_id": chain_result.token_id,
                                "tx_hash": chain_result.tx_hash,
                                "proof_url": chain_result.proof_url,
                                "value_before": live_metrics.score_value,
                                "upload_count": current_session.upload_count,
                                "proof_status": current_session.proof_status,
                                "mode": "emergency" if args.emergency_mode else "live_or_fallback",
                            }
                        )
                        chain_result.local_block_hash = local_block_hash
                        if not chain_result.tx_hash:
                            chain_result.tx_hash = "0x" + local_block_hash
                        print(f"[TouchGrass] Local immutable block #{chain_index}: {local_block_hash}")
                        latest_block_hash = local_block_hash
                        last_mint_at = now

                        current_session.nft_tx_hash = chain_result.tx_hash
                        current_session.token_id = chain_result.token_id
                        current_session.proof_url = chain_result.proof_url
                        current_session.metadata_path = chain_result.metadata_path
                        if chain_result.proof_url:
                            print(f"[TouchGrass] Public proof URL: {chain_result.proof_url}")
                        if chain_result.metadata_path:
                            print(f"[TouchGrass] Local metadata: {chain_result.metadata_path}")
                        latest_proof_url = chain_result.proof_url
                        chain_badge = chain_result.badge
                        live_metrics = ledger.record_verified_session(
                            current_session,
                            wallet_address=args.wallet_address,
                            chain_result=chain_result,
                        )
                        with upload_state.lock:
                            upload_state.latest_token_id = live_metrics.token_id
                            upload_state.latest_proof_url = chain_result.proof_url
                            upload_state.latest_flower_level = max(1, live_metrics.outdoor_sessions + 1)
                            upload_state.latest_value = live_metrics.score_value
                            upload_state.latest_block_hash = chain_result.local_block_hash
                            upload_state.latest_chain_badge = _display_chain_badge(chain_badge.value)
                        set_state(RecoveryAppState.MINTED)
                        mint_flash_until = now + 6.0
                        result_state_until = now + args.proof_result_seconds
                        with upload_state.lock:
                            upload_state.active_session_id = ""
                        current_session = None
                        qr_tile = make_qr_tile(upload_root_url, size_px=240)
                        cv2.imwrite(args.overlay_qr_file, qr_tile)
                        if chain_result.proof_url:
                            status_message = (
                                f"Proof verified. Local mint + IPFS proof ready. Value=${live_metrics.score_value:.2f}"
                            )
                        else:
                            status_message = (
                                f"Proof verified. Block {chain_result.local_block_hash[:10]}.. Value=${live_metrics.score_value:.2f}"
                            )
                        status_message_until = now + args.proof_result_seconds
                    else:
                        current_session.proof_status = "rejected"
                        status_message = (
                            "Upload fail: try a brighter outdoor shot with visible plants "
                            f"(veg={upload_signals.vegetation_score:.2f}, out={upload_signals.outdoor_score:.2f})."
                        )
                        status_message_until = now + 5.0

            if app_state in (RecoveryAppState.PROOF_VERIFIED, RecoveryAppState.MINTED) and now > result_state_until:
                look_started_at = None
                set_state(RecoveryAppState.ONLINE)

            if status_message and now > status_message_until:
                status_message = None
            if app_state == RecoveryAppState.OUTSIDE_MODE and status_message is None:
                status_message = "Outside mode active. Leave screen, then upload proof from session QR."
                status_message_until = now + 0.5

            if now - last_overlay_status_write >= 0.2:
                with upload_state.lock:
                    upload_state.latest_chain_badge = _display_chain_badge(chain_badge.value)
                    upload_state.latest_flower_level = max(1, live_metrics.outdoor_sessions + 1)
                    upload_state.latest_value = live_metrics.score_value
                    upload_state.latest_token_id = live_metrics.token_id
                _write_overlay_status(
                    args.overlay_status_file,
                    {
                        "ts": now,
                        "state": app_state.value,
                        "chain_badge": _display_chain_badge(chain_badge.value),
                        "face_detected": signals.face_detected,
                        "eyes_detected": signals.eyes_detected,
                        "looking_score": round(signals.looking_score, 4),
                        "screen_time_seconds": round(screen_time_seconds, 2),
                        "focus_streak_seconds": round(focus_streak_seconds, 2),
                        "status_message": status_message or "",
                        "demo_step": _demo_step_text(app_state) if args.demo_mode else "",
                        "flower_level": max(1, live_metrics.outdoor_sessions + 1),
                        "health_score": int(np.clip(56 + (7 * live_metrics.outdoor_sessions) - (8 * live_metrics.decay_penalties), 0, 100)),
                        "value_score": round(live_metrics.score_value, 2),
                        "sessions": live_metrics.outdoor_sessions,
                        "streak_days": live_metrics.streak_days,
                        "decay_penalties": live_metrics.decay_penalties,
                        "token_id": live_metrics.token_id,
                        "tx_hash": live_metrics.latest_tx_hash,
                        "proof_url": latest_proof_url,
                        "local_block_hash": latest_block_hash,
                        "last_mint_at": last_mint_at,
                        "mint_flash_until": mint_flash_until,
                        "qr_file": args.overlay_qr_file,
                    },
                )
                last_overlay_status_write = now

            debug_frame = draw_debug_overlay(
                frame_bgr=frame,
                signals=signals,
                app_state=app_state,
                chain_badge=chain_badge,
                live_metrics=live_metrics,
                session=current_session,
                prompt_visible=prompt_visible,
                screen_time_seconds=screen_time_seconds,
                focus_streak_seconds=focus_streak_seconds,
                flower_assets=flower_assets,
                mint_flash_until=mint_flash_until,
                latest_proof_url=latest_proof_url,
                qr_tile=qr_tile,
                status_message=status_message,
                demo_step_text=_demo_step_text(app_state) if args.demo_mode else "",
            )
            cv2.imshow(window_name, debug_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if (
                prompt_visible
                and args.auto_recovery_start
                and app_state == RecoveryAppState.RECOVERY_REQUIRED
                and (now - state_started_at) > 0.8
            ):
                start_recovery_session(now, signals, allow_no_face=True)
            if prompt_visible and key in (ord("y"), ord("Y"), 13, 32):
                start_recovery_session(now, signals, allow_no_face=args.demo_mode)
            elif key in (ord("r"), ord("R")):
                # Demo escape hatch: force start recovery flow even if prompt focus is missed.
                started = start_recovery_session(now, signals, allow_no_face=True)
                if not started:
                    status_message = "Recovery already active or unavailable."
                    status_message_until = now + 2.0
            elif prompt_visible and key in (ord("n"), ord("N")):
                prompt_visible = False
                look_started_at = now - (args.warning_hold_seconds * 0.5)
                status_message = "Recovery deferred. Warning loop continues."
                status_message_until = now + 3.0
            elif key == ord("g"):
                tracker.add_ml_sample(frame, is_vegetation=True)
                pos_n, neg_n, out_pos, out_neg = tracker.ml_sample_counts()
                status_message = f"Saved grass sample. veg:{pos_n}/{neg_n} outdoor:{out_pos}/{out_neg}"
                status_message_until = now + 2.5
            elif key == ord("b"):
                tracker.add_ml_sample(frame, is_vegetation=False)
                pos_n, neg_n, out_pos, out_neg = tracker.ml_sample_counts()
                status_message = f"Saved not-grass sample. veg:{pos_n}/{neg_n} outdoor:{out_pos}/{out_neg}"
                status_message_until = now + 2.5
            elif key == ord("o"):
                tracker.add_outdoor_sample(frame, is_outdoor=True)
                pos_n, neg_n, out_pos, out_neg = tracker.ml_sample_counts()
                status_message = f"Saved outdoor sample. veg:{pos_n}/{neg_n} outdoor:{out_pos}/{out_neg}"
                status_message_until = now + 2.5
            elif key == ord("i"):
                tracker.add_outdoor_sample(frame, is_outdoor=False)
                pos_n, neg_n, out_pos, out_neg = tracker.ml_sample_counts()
                status_message = f"Saved indoor sample. veg:{pos_n}/{neg_n} outdoor:{out_pos}/{out_neg}"
                status_message_until = now + 2.5
    finally:
        _write_overlay_stage(args.overlay_stage_file, RecoveryAppState.ONLINE)
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
