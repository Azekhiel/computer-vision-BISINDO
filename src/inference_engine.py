import json
import os
import queue
import shutil
import subprocess
import threading
import time
from collections import deque

import cv2
import faiss
import mediapipe as mp
import numpy as np

try:
    import torch
    TORCH_IMPORT_ERROR = None
except Exception as exc:  # Jetson CUDA wheels can fail during import if CUDA deps are missing.
    torch = None
    TORCH_IMPORT_ERROR = exc

import faiss_manager as fm
import feature_engine as fe

lm = None
sgm = None
tm = None

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
FAISS_INDEX = os.path.join(MODEL_DIR, "sign_language.index")
FAISS_LABELS = os.path.join(MODEL_DIR, "label_map.npy")
LSTM_WEIGHTS = os.path.join(MODEL_DIR, "lstm_weights.pth")
LSTM_LABELS = os.path.join(MODEL_DIR, "lstm_labels.json")
TRANSFORMER_WEIGHTS = os.path.join(MODEL_DIR, "transformer_weights.pth")
TRANSFORMER_LABELS = os.path.join(MODEL_DIR, "transformer_labels.json")
SEGMENTER_WEIGHTS = os.path.join(MODEL_DIR, "segmenter_weights.pth")
SEGMENTER_METADATA = os.path.join(MODEL_DIR, "segmenter_metadata.json")
LSTM_METADATA = os.path.join(MODEL_DIR, "lstm_metadata.json")
TRANSFORMER_METADATA = os.path.join(MODEL_DIR, "transformer_metadata.json")

VAD_SOURCE_WINDOW = 30
VAD_TARGET_FRAMES = 30
VAD_START_CONFIDENCE = 0.65
VAD_STOP_CONFIDENCE = 0.45
VAD_EMA_ALPHA = 0.35
VAD_START_HITS = 2
VAD_STOP_HITS = 8
PRE_ROLL_FRAMES = 20
MIN_VALID_FRAMES = 8
MAX_RECORD_FRAMES = 150
SENTENCE_IDLE_SECONDS = 4.0
PYTORCH_CONFIDENCE = 0.65

LIVE_START_THRESH = 0.015
LIVE_STOP_THRESH = 0.008
LIVE_TRIM_PAD = 2


class EspeakSpeaker:
    """Small non-blocking Linux-safe TTS worker."""

    def __init__(self):
        self.binary = shutil.which("espeak-ng") or shutil.which("espeak")
        self.queue: queue.Queue[str | None] = queue.Queue()
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()

    def start(self):
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def say(self, text: str):
        if text and self.binary:
            self.queue.put(text)

    def stop(self):
        self.stop_event.set()
        self.queue.put(None)

    def _loop(self):
        while not self.stop_event.is_set():
            text = self.queue.get()
            if text is None:
                break
            if not self.binary:
                continue
            cmd = [self.binary, "-s", "150"]
            if os.path.basename(self.binary) == "espeak-ng":
                cmd += ["-v", "id"]
            cmd.append(text)
            try:
                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=12,
                    check=False,
                )
            except Exception:
                pass


def _torch_device(use_cpu=False):
    if torch is None:
        raise RuntimeError(_torch_unavailable_message())
    return torch.device("cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))


def _torch_unavailable_message():
    if TORCH_IMPORT_ERROR is None:
        return "PyTorch tidak tersedia."
    return (
        "PyTorch gagal di-import. Pada Jetson, ini biasanya karena wheel CUDA tidak cocok "
        f"atau dependency hilang: {TORCH_IMPORT_ERROR}"
    )


def _load_torch_module(name: str):
    global lm, sgm, tm
    if torch is None:
        raise RuntimeError(_torch_unavailable_message())
    if name == "segmenter":
        if sgm is None:
            import segmenter_manager as _sgm
            sgm = _sgm
        return sgm
    if name == "lstm":
        if lm is None:
            import lstm_manager as _lm
            lm = _lm
        return lm
    if name == "transformer":
        if tm is None:
            import transformer_manager as _tm
            tm = _tm
        return tm
    raise ValueError(f"Unknown torch module: {name}")


def _metadata_schema(path: str) -> str:
    if not os.path.exists(path):
        return fe.LEGACY_SCHEMA
    try:
        with open(path, "r") as f:
            return str(json.load(f).get("feature_schema", fe.LEGACY_SCHEMA))
    except Exception:
        return fe.LEGACY_SCHEMA


def load_segmenter_model(use_cpu=False):
    if torch is None:
        return None, None, _torch_unavailable_message()
    device = _torch_device(use_cpu)
    if not os.path.exists(SEGMENTER_WEIGHTS):
        return None, device, "Model Segmenter belum dilatih. Jalankan Tahap 1 di UI."
    schema = _metadata_schema(SEGMENTER_METADATA)
    if schema != fe.FEATURE_SCHEMA:
        return None, device, f"Model Segmenter stale ({schema}). Retrain untuk {fe.FEATURE_SCHEMA}."

    segmenter_module = _load_torch_module("segmenter")
    model = segmenter_module.VADSegmenterModel(input_dim=179, hidden_dim=64)
    model.load_state_dict(torch.load(SEGMENTER_WEIGHTS, map_location=device))
    model.to(device).eval()
    return model, device, "OK"


def load_classifier_model(model_type, use_cpu=False):
    if model_type == "faiss":
        if not os.path.exists(FAISS_INDEX) or not os.path.exists(FAISS_LABELS):
            return None, None, None, "Index FAISS belum di-build."
        metadata = fm.load_faiss_metadata()
        if metadata.get("feature_schema") != fe.FEATURE_SCHEMA:
            return (
                None,
                None,
                None,
                f"Index FAISS stale ({metadata.get('feature_schema')}). Rebuild FAISS untuk {fe.FEATURE_SCHEMA}.",
            )
        model = faiss.read_index(FAISS_INDEX)
        labels = np.load(FAISS_LABELS, allow_pickle=True)
        return model, labels, None, "OK"

    if torch is None:
        return None, None, None, _torch_unavailable_message()

    device = _torch_device(use_cpu)
    weights_path = LSTM_WEIGHTS if model_type == "lstm" else TRANSFORMER_WEIGHTS
    labels_path = LSTM_LABELS if model_type == "lstm" else TRANSFORMER_LABELS
    metadata_path = LSTM_METADATA if model_type == "lstm" else TRANSFORMER_METADATA
    if not os.path.exists(weights_path) or not os.path.exists(labels_path):
        return None, None, None, f"Model {model_type.upper()} belum dilatih."
    schema = _metadata_schema(metadata_path)
    if schema != fe.FEATURE_SCHEMA:
        return None, None, None, f"Model {model_type.upper()} stale ({schema}). Retrain untuk {fe.FEATURE_SCHEMA}."

    with open(labels_path, "r") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    num_classes = len(label_map)
    if model_type == "lstm":
        lstm_module = _load_torch_module("lstm")
        model = lstm_module.BiLSTMAttentionModel(input_dim=179, hidden_dim=256, num_classes=num_classes, num_layers=2)
    else:
        transformer_module = _load_torch_module("transformer")
        model = transformer_module.TransformerSignModel(
            input_dim=179,
            d_model=256,
            nhead=8,
            num_layers=3,
            dim_feedforward=512,
            num_classes=num_classes,
        )

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model, label_map, device, "OK"


def _build_vad_tensor(frame_buffer, device):
    builder = fe.SequenceBuilder()
    for observation in frame_buffer:
        builder.add_observation(observation)
    seq_list, _ = builder.build()
    if not seq_list:
        return None
    seq = np.asarray(seq_list, dtype=np.float32)
    std_seq = fm.interpolate_sequence(seq, VAD_TARGET_FRAMES).astype(np.float32)
    return torch.from_numpy(std_seq).unsqueeze(0).to(device)


def _trim_live_sequence(sequence, scores):
    if not sequence or len(sequence) < 5:
        return sequence

    n = len(sequence)
    start_idx, end_idx = 0, n - 1
    for i, score in enumerate(scores):
        if score > LIVE_START_THRESH:
            start_idx = max(0, i - LIVE_TRIM_PAD)
            break
    for i in range(n - 1, -1, -1):
        if scores[i] > LIVE_STOP_THRESH:
            end_idx = min(n - 1, i + LIVE_TRIM_PAD)
            break
    if start_idx >= end_idx:
        return sequence
    return sequence[start_idx : end_idx + 1]


class LiveInferenceWorker(threading.Thread):
    """
    Background inference loop for Tkinter.

    Tkinter never gets touched from this thread. Status updates are pushed into
    status_queue and the UI polls them with root.after().
    """

    def __init__(self, selected_model="faiss", mp_device="CPU", status_queue=None, camera_index=0):
        super().__init__(daemon=True)
        self.selected_model = selected_model
        self.mp_device = mp_device
        self.status_queue = status_queue or queue.Queue()
        self.camera_index = camera_index
        self.stop_event = threading.Event()
        self.speaker = EspeakSpeaker()

    def stop(self):
        self.stop_event.set()
        self.speaker.stop()

    def _emit(self, event="status", **payload):
        payload["event"] = event
        self.status_queue.put(payload)

    def run(self):
        try:
            ok, msg = self._run_loop()
        except Exception as exc:
            ok, msg = False, f"Live inference error: {exc}"
        self._emit("done", ok=ok, message=msg)

    def _run_loop(self):
        print(f"\n--- Memulai Smart Two-Stage Inference: {self.selected_model.upper()} ---")
        use_cpu_for_ai = self.mp_device == "CPU"

        segmenter, device_seg, msg_seg = load_segmenter_model(use_cpu_for_ai)
        if segmenter is None:
            return False, msg_seg

        classifier, label_map, device_cls, msg_cls = load_classifier_model(self.selected_model, use_cpu_for_ai)
        if classifier is None:
            return False, msg_cls

        self.speaker.start()

        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            self.speaker.stop()
            return False, "Kamera tidak dapat dibuka."

        vad_buffer = deque(maxlen=VAD_SOURCE_WINDOW)
        pre_roll = deque(maxlen=PRE_ROLL_FRAMES)
        builder = fe.SequenceBuilder()
        combo_buffer: list[str] = []
        is_recording_word = False
        active_hits = 0
        idle_hits = 0
        vad_prob = 0.0
        vad_prob_ema = 0.0
        last_word_time = time.time()
        current_prediction = "SIAP. SILAKAN BERGERAK."
        last_emit = 0.0

        try:
            with mp.solutions.holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.35,
                smooth_landmarks=True,
                model_complexity=0,
            ) as holistic:
                while not self.stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.resize(frame, (640, 480))
                    _, w, _ = frame.shape
                    results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    observation = fe.extract_frame_observation(results)
                    vad_buffer.append(observation)
                    pre_roll.append(observation)

                    if len(vad_buffer) == VAD_SOURCE_WINDOW:
                        seg_input = _build_vad_tensor(vad_buffer, device_seg)
                        if seg_input is not None:
                            with torch.inference_mode():
                                seg_out = segmenter(seg_input)
                                vad_prob = float(torch.sigmoid(seg_out).item())
                            vad_prob_ema = (
                                vad_prob
                                if vad_prob_ema <= 0.0
                                else VAD_EMA_ALPHA * vad_prob + (1.0 - VAD_EMA_ALPHA) * vad_prob_ema
                            )

                    start_active = vad_prob_ema >= VAD_START_CONFIDENCE
                    stop_idle = vad_prob_ema <= VAD_STOP_CONFIDENCE
                    active_hits = active_hits + 1 if start_active else 0
                    idle_hits = idle_hits + 1 if stop_idle else 0

                    if not is_recording_word and active_hits >= VAD_START_HITS:
                        is_recording_word = True
                        idle_hits = 0
                        builder.reset()
                        for buffered_sample in pre_roll:
                            builder.add_observation(buffered_sample)
                        current_prediction = "MEREKAM KATA..."

                    elif is_recording_word:
                        builder.add_observation(observation)
                        should_finalize = idle_hits >= VAD_STOP_HITS or len(builder._vectors) >= MAX_RECORD_FRAMES
                        if should_finalize:
                            is_recording_word = False
                            active_hits = 0
                            idle_hits = 0
                            label_text = self._finalize_word(builder, classifier, label_map, device_cls)
                            current_prediction = label_text
                            if label_text.startswith("+ "):
                                combo_buffer.append(label_text[2:])
                                last_word_time = time.time()
                            builder.reset()

                    elif combo_buffer and (time.time() - last_word_time) >= SENTENCE_IDLE_SECONDS:
                        sentence = " ".join(combo_buffer)
                        current_prediction = f"KALIMAT: {sentence}"
                        self.speaker.say(sentence.replace("_", " "))
                        combo_buffer = []
                        last_word_time = time.time()

                    self._render_overlay(frame, w, current_prediction, is_recording_word, vad_prob_ema, combo_buffer)
                    cv2.imshow("BISINDO Live Translator", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                    now = time.time()
                    if now - last_emit >= 0.25:
                        self._emit(
                            prediction=current_prediction,
                            vad_probability=vad_prob_ema,
                            recording=is_recording_word,
                            words=list(combo_buffer),
                        )
                        last_emit = now

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.speaker.stop()

        return True, "Inferensi selesai."

    def _finalize_word(self, builder, classifier, label_map, device_cls):
        seq_list, scores = builder.build()
        seq_list = _trim_live_sequence(seq_list, scores)
        if len(seq_list) < MIN_VALID_FRAMES:
            return "GERAKAN TERLALU PENDEK"

        seq_array = np.asarray(seq_list, dtype=np.float32)

        if self.selected_model == "faiss":
            try:
                pred_label, score, _, _ = fm.search_sequence(classifier, label_map, seq_array[:, :176])
            except ValueError as exc:
                return f"REBUILD FAISS: {exc}"
            if pred_label != "unknown":
                return f"+ {pred_label.upper()}"
            return f"TIDAK DIKENAL ({score:.2f})"

        tensor_seq = torch.from_numpy(seq_array).unsqueeze(0).to(device_cls)
        tensor_len = torch.tensor([len(seq_array)], device=device_cls)
        with torch.inference_mode():
            outputs = classifier(tensor_seq, tensor_len)
            probs = torch.softmax(outputs, dim=1)
            conf, idx = torch.max(probs, 1)

        if float(conf.item()) > PYTORCH_CONFIDENCE:
            pred_label = label_map[int(idx.item())]
            return f"+ {pred_label.upper()}"
        return f"TIDAK YAKIN ({conf.item():.2f})"

    @staticmethod
    def _render_overlay(frame, width, current_prediction, is_recording_word, vad_prob, combo_buffer):
        bar_color = (0, 0, 255) if is_recording_word else (0, 255, 0)
        seg_w = int(np.clip(vad_prob, 0.0, 1.0) * 200)
        cv2.rectangle(frame, (20, 80), (20 + seg_w, 95), bar_color, -1)
        cv2.rectangle(frame, (20, 80), (220, 95), (255, 255, 255), 1)
        cv2.putText(
            frame,
            f"SATPAM/VAD: {vad_prob * 100:.1f}%",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        header_color = (0, 165, 255) if is_recording_word else (245, 117, 16)
        cv2.rectangle(frame, (0, 0), (width, 60), header_color, -1)
        cv2.putText(
            frame,
            f"STATUS: {current_prediction}",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
        )

        y_pos = 150
        if combo_buffer:
            cv2.putText(
                frame,
                f"Isi Buffer ({len(combo_buffer)} kata tersimpan):",
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            y_pos += 25
            kata_list = " - ".join(combo_buffer)
            cv2.putText(frame, kata_list, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)


def start_live_inference(selected_model="faiss", mp_device="CPU", status_queue=None):
    worker = LiveInferenceWorker(selected_model=selected_model, mp_device=mp_device, status_queue=status_queue)
    worker.start()
    return worker


def run_live_inference(selected_model="faiss", mp_device="CPU"):
    """Backward-compatible blocking entrypoint for direct script usage."""
    status_queue = queue.Queue()
    worker = start_live_inference(selected_model, mp_device, status_queue)
    worker.join()
    ok, msg = True, "Inferensi selesai."
    while not status_queue.empty():
        item = status_queue.get()
        if item.get("event") == "done":
            ok = bool(item.get("ok", True))
            msg = item.get("message", msg)
    return ok, msg
