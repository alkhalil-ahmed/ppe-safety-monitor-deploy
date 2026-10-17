"""ALTAVE Site Safety Intelligence - Streamlit PPE monitoring dashboard."""

from __future__ import annotations

import tempfile
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any
from collections import deque

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


APP_TITLE = "PPE Safety Monitor"
MODEL_PATH = Path(__file__).with_name("best (4).pt")
PROFILE_IMAGE_PATH = Path(__file__).with_name("Alkhalil Ahmed (2).jpg")
PERSON = "person"
VIOLATION_CLASSES = {"no-hardhat", "no-safety-vest"}
REQUIRED_PPE_CLASSES = {"hardhat", "safety-vest"}
DISPLAY_NAMES = {
    "no-hardhat": "Worker Unhelmeted",
    "no-safety-vest": "Missing Safety Vest",
}
CLASS_COLORS = {
    "Hardhat": (80, 220, 255),
    "Safety-Vest": (80, 220, 255),
    "Person": (220, 100, 255),
    "NO-Hardhat": (80, 80, 255),
    "NO-Safety-Vest": (80, 80, 255),
}
LIVE_VIOLATION_LOG: deque[dict[str, str]] = deque(maxlen=250)
LIVE_LOG_LOCK = Lock()
LIVE_STATUS_LOCK = Lock()
VIOLATION_CONFIRMATION_SECONDS = 1.0


class ViolationStatusTracker:
    """Require a continuous violation for the configured confirmation period."""

    def __init__(self) -> None:
        self.pending_since: float | None = None
        self.confirmed = False

    def update(self, has_violation: bool, now: float | None = None) -> bool:
        current_time = time.monotonic() if now is None else now
        if not has_violation:
            self.pending_since = None
            self.confirmed = False
            return False
        if self.pending_since is None:
            self.pending_since = current_time
        if current_time - self.pending_since >= VIOLATION_CONFIRMATION_SECONDS:
            self.confirmed = True
        return self.confirmed


@st.cache_resource(show_spinner="Loading PPE detection engine...")
def load_model() -> YOLO:
    """Load the bundled model once per Streamlit process."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file was not found: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))


@st.cache_resource
def get_live_status_state() -> dict[str, bool]:
    return {"violation": False}


def normalise_class_name(name: str) -> str:
    """Normalise model labels while retaining the configured display names."""
    return name.strip().replace("_", "-")


def canonical_class_name(name: str) -> str:
    return normalise_class_name(name).casefold()


def box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def fit_frame_for_display(
    frame: np.ndarray, max_width: int = 1400, max_height: int = 720
) -> np.ndarray:
    """Scale a frame for the browser while preserving its aspect ratio."""
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return frame
    return cv2.resize(
        frame,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def intersection_area(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> int:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    return box_area((x1, y1, x2, y2))


def is_associated_with_person(
    violation_box: tuple[int, int, int, int], person_box: tuple[int, int, int, int]
) -> bool:
    """Return true when a violation overlaps a person or its centre is inside it."""
    overlap = intersection_area(violation_box, person_box)
    if overlap == 0:
        return False
    violation_area = box_area(violation_box)
    person_area = box_area(person_box)
    if violation_area == 0 or person_area == 0:
        return False
    # A small intersection can be a detector edge-touch. Require meaningful overlap,
    # or accept a centre point inside the worker box.
    centre_x = (violation_box[0] + violation_box[2]) / 2
    centre_y = (violation_box[1] + violation_box[3]) / 2
    centre_inside = (
        person_box[0] <= centre_x <= person_box[2]
        and person_box[1] <= centre_y <= person_box[3]
    )
    return centre_inside or overlap / min(violation_area, person_area) >= 0.15


def extract_detections(result: Any, class_names: dict[int, str] | list[str]) -> list[dict[str, Any]]:
    """Convert an Ultralytics result into application-owned detection records."""
    detections: list[dict[str, Any]] = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        raw_name = class_names[cls_id]
        name = normalise_class_name(str(raw_name))
        coordinates = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = [int(value) for value in coordinates]
        detections.append(
            {
                "name": name,
                "canonical_name": canonical_class_name(name),
                "box": (x1, y1, x2, y2),
                "confidence": float(box.conf[0].item()),
            }
        )
    return detections


def draw_hud_box(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    colour: tuple[int, int, int],
    label: str,
) -> None:
    """Draw a compact corner-bracket overlay with a subtle label chip."""
    x1, y1, x2, y2 = box
    height, width = frame.shape[:2]
    x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
    y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return

    stroke = 2
    corner_length = max(12, min(28, int(min(x2 - x1, y2 - y1) * 0.22)))
    segments = (
        ((x1, y1), (x1 + corner_length, y1)),
        ((x1, y1), (x1, y1 + corner_length)),
        ((x2, y1), (x2 - corner_length, y1)),
        ((x2, y1), (x2, y1 + corner_length)),
        ((x1, y2), (x1 + corner_length, y2)),
        ((x1, y2), (x1, y2 - corner_length)),
        ((x2, y2), (x2 - corner_length, y2)),
        ((x2, y2), (x2, y2 - corner_length)),
    )
    for start, end in segments:
        cv2.line(frame, start, end, colour, stroke, cv2.LINE_AA)

    text = label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.46
    text_thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, text_thickness
    )
    label_x = x1
    label_y = y1 - 7
    if label_y - text_height - baseline < 0:
        label_y = min(height - baseline - 4, y1 + text_height + baseline + 7)
    label_x = min(label_x, max(0, width - text_width - 12))
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (label_x, label_y - text_height - baseline - 4),
        (label_x + text_width + 10, label_y + 3),
        (8, 10, 20),
        thickness=-1,
    )
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.putText(
        frame,
        text,
        (label_x + 5, label_y),
        font,
        font_scale,
        (240, 248, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def analyse_frame(
    model: YOLO,
    frame: np.ndarray,
    status_tracker: ViolationStatusTracker | None = None,
    elapsed_seconds: float | None = None,
) -> tuple[np.ndarray, list[dict[str, str]], int]:
    """Detect objects, associate configured violations, and render label-only boxes."""
    class_names = model.names
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = model.predict(source=rgb_frame, conf=0.25, verbose=False)[0]
    detections = extract_detections(result, class_names)
    persons = [
        item for item in detections if item["canonical_name"] == canonical_class_name(PERSON)
    ]
    violations: list[dict[str, str]] = []
    person_statuses: dict[int, str] = {}

    for person_index, person in enumerate(persons):
        associated_names = {
            item["canonical_name"]
            for item in detections
            if item is not person
            and is_associated_with_person(item["box"], person["box"])
        }
        missing_ppe = associated_names & VIOLATION_CLASSES
        has_required_ppe = REQUIRED_PPE_CLASSES <= associated_names
        if missing_ppe or not has_required_ppe:
            person_statuses[person_index] = "NOT SAFE"
            missing_classes = set(missing_ppe)
            if "hardhat" not in associated_names:
                missing_classes.add("no-hardhat")
            if "safety-vest" not in associated_names:
                missing_classes.add("no-safety-vest")
            for missing_class in sorted(missing_classes):
                violations.append(
                    {
                        "type": DISPLAY_NAMES[missing_class],
                        "severity": "HIGH",
                        "status": "Logged / Alert Sent",
                    }
                )
        elif has_required_ppe:
            person_statuses[person_index] = "SAFE"

    has_worker_violation = any(status == "NOT SAFE" for status in person_statuses.values())
    violation_confirmed = (
        status_tracker.update(has_worker_violation, elapsed_seconds)
        if status_tracker is not None
        else bool(violations)
    )
    rendered = frame.copy()
    for detection_index, item in enumerate(detections):
        colour = CLASS_COLORS.get(item["name"], (255, 255, 255))
        if item["canonical_name"] == canonical_class_name(PERSON):
            person_index = sum(
                1
                for prior in detections[:detection_index]
                if prior["canonical_name"] == canonical_class_name(PERSON)
            )
            person_status = person_statuses.get(person_index, "CHECKING")
            if person_status == "SAFE":
                colour = (0, 255, 0)
            elif person_status == "NOT SAFE" and violation_confirmed:
                colour = (0, 0, 255)
            else:
                colour = (0, 200, 255)
        else:
            person_index = -1
            person_status = ""
        label = item["name"]
        if item["canonical_name"] == canonical_class_name(PERSON):
            label = f"Person [{person_status}]"
        draw_hud_box(
            rendered,
            item["box"],
            colour,
            label,
        )
    return rendered, violations if violation_confirmed else [], len(persons)


def add_violation_log(violations: list[dict[str, str]], timestamp: str) -> None:
    for violation in violations:
        st.session_state.violation_log.append(
            {
                "Timestamp": timestamp,
                "Violation Type": violation["type"],
                "Severity": violation["severity"],
                "Action Status": violation["status"],
            }
        )


def render_alert(violations: list[dict[str, str]]) -> None:
    if violations:
        st.markdown(
            '<div class="violation-banner">🚨 SAFETY VIOLATION DETECTED</div>',
            unsafe_allow_html=True,
        )


def render_status_light(is_violation: bool, container: Any = st) -> None:
    colour = "#ff1f3d" if is_violation else "#10b981"
    status = "VIOLATION" if is_violation else "COMPLIANT"
    container.markdown(
        f"""
        <style>
            :root {{
                --ppe-alert-glow: {"rgba(239, 24, 56, 0.42)" if is_violation else "rgba(0, 229, 255, 0.08)"};
            }}
            html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
                background:
                    radial-gradient(circle at 78% 18%, var(--ppe-alert-glow), transparent 42rem),
                    radial-gradient(circle at 50% 55%, {"rgba(239, 24, 56, 0.18)" if is_violation else "transparent"}, transparent 48rem),
                    radial-gradient(circle at 12% 4%, rgba(0, 229, 255, 0.06), transparent 25rem),
                    linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px),
                    #0b0f19 !important;
                background-size: auto, auto, 34px 34px, 34px 34px, auto !important;
                transition: background 300ms ease;
            }}
        </style>
        <div class="status-light-card">
            <div class="lamp-housing">
                <span class="status-light" style="--lamp-colour:{colour}; background:{colour};"></span>
            </div>
            <span><strong>{status}</strong><small>PPE STATUS</small></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def set_live_status(is_violation: bool) -> None:
    with LIVE_STATUS_LOCK:
        get_live_status_state()["violation"] = is_violation


def get_live_status() -> bool:
    with LIVE_STATUS_LOCK:
        return get_live_status_state()["violation"]


@st.fragment(run_every="500ms")
def render_live_status_light() -> None:
    render_status_light(get_live_status())


def reset_live_status() -> None:
    set_live_status(False)


def render_kpis() -> None:
    log = st.session_state.violation_log
    with LIVE_LOG_LOCK:
        live_violation_count = len(LIVE_VIOLATION_LOG)
    workers = st.session_state.get("last_worker_count", 0)
    violation_count = len(log) + live_violation_count
    compliant = max(0, workers - violation_count)
    first, second, third = st.columns(3)
    first.metric("Total Workers Detected", workers)
    second.metric("Compliant Count", compliant)
    third.metric("Violations Count", violation_count)


def render_log() -> None:
    st.markdown(
        '<div class="terminal-heading"><span class="terminal-dot"></span>'
        'LIVE VIOLATION FEED <span class="terminal-state">STREAMING</span></div>',
        unsafe_allow_html=True,
    )
    with LIVE_LOG_LOCK:
        combined_log = [*st.session_state.violation_log, *LIVE_VIOLATION_LOG]
    if combined_log:
        st.dataframe(
            pd.DataFrame(combined_log),
            width="stretch",
            height=150,
            hide_index=True,
        )
    else:
        st.info("No configured violations have been logged. Detected objects are shown in the annotated output.")


def process_image(model: YOLO, uploaded_file: Any, status_container: Any) -> None:
    try:
        image = np.array(Image.open(uploaded_file).convert("RGB"))
        annotated, violations, worker_count = analyse_frame(
            model, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )
        st.session_state.last_worker_count = worker_count
        display_frame = fit_frame_for_display(annotated)
        st.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), width="stretch")
        render_status_light(bool(violations), status_container)
        timestamp = datetime.now().strftime("%H:%M:%S")
        add_violation_log(violations, timestamp)
        render_alert(violations)
        if violations:
            st.warning("Spatial check passed: the alert is associated with a detected Person box.")
    except (OSError, ValueError, cv2.error) as error:
        st.error(f"Unable to process the image: {error}")


def process_video(model: YOLO, uploaded_file: Any, status_container: Any) -> None:
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    input_path: str | None = None
    output_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as source:
            source.write(uploaded_file.getbuffer())
            input_path = source.name
        capture = cv2.VideoCapture(input_path)
        if not capture.isOpened():
            raise ValueError("The uploaded file is not a readable video.")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
        if width <= 0 or height <= 0:
            raise ValueError("The video has invalid dimensions.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as destination:
            output_path = destination.name
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise ValueError("The application could not create an output video.")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        progress = st.progress(0, text="Analysing video...")
        preview = st.empty()
        frame_index = 0
        status_tracker = ViolationStatusTracker()
        all_violations: list[dict[str, str]] = []
        while True:
            success, frame = capture.read()
            if not success:
                break
            elapsed_seconds = frame_index / fps
            annotated, violations, worker_count = analyse_frame(
                model, frame, status_tracker, elapsed_seconds
            )
            writer.write(annotated)
            display_frame = fit_frame_for_display(annotated)
            preview.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), width="stretch")
            timestamp = f"{int(frame_index / fps) // 60:02d}:{int(frame_index / fps) % 60:02d}"
            add_violation_log(violations, timestamp)
            all_violations.extend(violations)
            st.session_state.last_worker_count = worker_count
            frame_index += 1
            if frame_count:
                progress.progress(min(frame_index / frame_count, 1.0), text=f"Analysing frame {frame_index}/{frame_count}")
        capture.release()
        writer.release()
        progress.empty()
        if frame_index == 0:
            raise ValueError("No readable frames were found in the uploaded video.")
        st.video(output_path, width="stretch")
        render_status_light(status_tracker.confirmed, status_container)
        st.success(f"Video analysis complete: {frame_index} frames processed.")
        render_alert(all_violations)
    except (OSError, ValueError, cv2.error) as error:
        st.error(f"Unable to process the video: {error}")
    finally:
        if input_path:
            Path(input_path).unlink(missing_ok=True)


def render_live_camera(model: YOLO) -> None:
    try:
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
    except ImportError:
        st.error("Live camera mode requires the `streamlit-webrtc` dependency.")
        return

    lock = Lock()
    class PPEVideoProcessor:
        def __init__(self) -> None:
            self.model = model
            self.last_violations: list[dict[str, str]] = []
            self.status_tracker = ViolationStatusTracker()

        def recv(self, frame: Any) -> Any:
            image = frame.to_ndarray(format="bgr24")
            annotated, violations, _ = analyse_frame(
                self.model, image, self.status_tracker
            )
            set_live_status(self.status_tracker.confirmed)
            with lock:
                self.last_violations = violations
                for violation in violations:
                    LIVE_VIOLATION_LOG.append(
                        {
                            "Timestamp": datetime.now().strftime("%H:%M:%S"),
                            "Violation Type": violation["type"],
                            "Severity": violation["severity"],
                            "Action Status": violation["status"],
                        }
                    )
            return frame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="ppe-live-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PPEVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


def initialise_state() -> None:
    if "violation_log" not in st.session_state:
        st.session_state.violation_log = []
    if "last_worker_count" not in st.session_state:
        st.session_state.last_worker_count = 0


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🦺", layout="wide")
    initialise_state()
    st.markdown(
        """
        <style>
        :root {
            --navy: #10243e;
            --navy-soft: #1d3858;
            --ink: #172b45;
            --muted: #60738a;
            --line: #dbe4ee;
            --surface: #ffffff;
            --canvas: #f4f7fb;
            --amber: #e7a52b;
            --green: #16845b;
        }
        .stApp {
            background: var(--canvas);
            color: var(--ink);
        }
        .block-container {
            max-width: 1240px;
            padding: 3rem 3rem 4rem;
        }
        header[data-testid="stHeader"] {
            background: transparent;
        }
        .hero {
            background: linear-gradient(120deg, #10243e 0%, #1d3858 100%);
            border-radius: 18px;
            padding: 2rem 2.25rem 1.75rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 12px 28px rgba(16, 36, 62, 0.14);
        }
        .hero h1 {
            color: #ffffff;
            font-size: clamp(1.8rem, 3vw, 2.65rem);
            letter-spacing: -0.03em;
            line-height: 1.15;
            margin: 0 0 0.65rem;
        }
        .hero p {
            color: #c9d7e6;
            font-size: 0.98rem;
            margin: 0;
        }
        .status {
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 999px;
            color: #9af0c5;
            float: right;
            font-size: 0.88rem;
            font-weight: 700;
            margin-top: 0.35rem;
            padding: 0.55rem 0.9rem;
        }
        .section-label {
            color: var(--navy);
            font-size: 1.15rem;
            font-weight: 750;
            margin: 1.5rem 0 0.75rem;
        }
        [data-testid="stTabs"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 14px;
            box-shadow: 0 5px 18px rgba(16, 36, 62, 0.06);
            padding: 0.4rem 1.2rem 1.4rem;
        }
        [data-testid="stTabs"] button {
            color: var(--muted);
            font-weight: 650;
            padding: 0.9rem 1rem;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--navy);
        }
        [data-testid="stFileUploader"] {
            background: #f8fafc;
            border: 1px dashed #aebdcd;
            border-radius: 10px;
            margin-top: 0.4rem;
            padding: 0.45rem;
        }
        [data-testid="stFileUploader"] section {
            background: transparent;
            border: 0;
        }
        [data-testid="stImage"] {
            align-items: center;
            background: #eef3f8;
            border: 1px solid var(--line);
            border-radius: 12px;
            display: flex;
            justify-content: center;
            max-height: 72vh;
            overflow: hidden;
            padding: 0.65rem;
        }
        [data-testid="stImage"] img {
            max-height: 68vh;
            max-width: 100%;
            object-fit: contain;
            width: auto;
        }
        [data-testid="stVideo"] {
            background: #0d1b2a;
            border: 1px solid var(--line);
            border-radius: 12px;
            max-height: 72vh;
            overflow: hidden;
            padding: 0.65rem;
        }
        [data-testid="stVideo"] video {
            display: block;
            margin: 0 auto;
            max-height: 68vh;
            max-width: 100%;
            object-fit: contain;
        }
        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-top: 4px solid var(--amber);
            border-radius: 12px;
            box-shadow: 0 5px 18px rgba(16, 36, 62, 0.06);
            padding: 1rem 1.15rem;
        }
        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-size: 0.83rem;
            font-weight: 700;
        }
        [data-testid="stMetricValue"] {
            color: var(--navy);
            font-size: 2rem;
            font-weight: 750;
        }
        [data-testid="stDataFrame"], [data-testid="stAlert"] {
            border-radius: 10px;
        }
        [data-testid="stAlert"] {
            border: 1px solid var(--line);
        }
        .stButton > button {
            border: 1px solid #c5d1de;
            border-radius: 8px;
            color: var(--navy);
            font-weight: 650;
        }
        .stButton > button:hover {
            border-color: var(--navy-soft);
            color: var(--navy-soft);
        }
        .stCaption, [data-testid="stCaptionContainer"] {
            color: var(--muted);
        }
        /* Industrial dashboard overrides */
        .stApp, [data-testid="stAppViewContainer"] {
            background: #0e1117;
            color: #f1f4f8;
        }
        .block-container {
            max-width: 1500px;
            padding: 1.4rem 2rem 2rem;
        }
        [data-testid="stHeader"] {
            background: #0e1117;
        }
        .dashboard-header {
            align-items: center;
            border-bottom: 1px solid #263241;
            display: flex;
            justify-content: space-between;
            margin-bottom: 1.2rem;
            padding-bottom: 1rem;
        }
        .dashboard-header h1 {
            color: #ffffff;
            font-size: clamp(1.45rem, 2.5vw, 2.15rem);
            letter-spacing: -0.03em;
            margin: 0;
        }
        .dashboard-header p {
            color: #9daabd;
            margin: 0.35rem 0 0;
        }
        .live-pill {
            background: rgba(40, 167, 69, 0.14);
            border: 1px solid #28a745;
            border-radius: 999px;
            color: #75e394;
            font-size: 0.8rem;
            font-weight: 700;
            padding: 0.5rem 0.8rem;
            white-space: nowrap;
        }
        .dashboard-card {
            background: #151a22;
            border: 1px solid #293545;
            border-radius: 12px;
            padding: 1rem 1.1rem;
        }
        .card-title {
            color: #ffc107;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            margin-bottom: 0.85rem;
            text-transform: uppercase;
        }
        .profile-card {
            align-items: center;
            background: linear-gradient(145deg, #182231, #11161e);
            border: 1px solid #39485a;
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            padding: 1.1rem;
            text-align: center;
        }
        .profile-name {
            color: #ffffff;
            font-size: 1.12rem;
            font-weight: 850;
            letter-spacing: 0.05em;
            margin-top: 0.7rem;
        }
        .profile-subtitle {
            color: #aeb9c8;
            font-size: 0.78rem;
            margin-top: 0.25rem;
        }
        [data-testid="stImage"] {
            background: #10151d;
            border: 1px solid #344253;
            border-radius: 10px;
        }
        [data-testid="stMetric"] {
            background: #151a22;
            border: 1px solid #293545;
            border-left: 4px solid #ffc107;
            border-radius: 8px;
            box-shadow: none;
            padding: 0.7rem 0.8rem;
        }
        [data-testid="stMetricLabel"] {
            color: #aeb9c8;
        }
        [data-testid="stMetricValue"] {
            color: #ffffff;
            font-size: 1.65rem;
        }
        [data-testid="stFileUploader"] {
            background: #10151d;
            border-color: #465568;
        }
        [data-testid="stTabs"], .stRadio {
            color: #f1f4f8;
        }
        [data-testid="stImage"], [data-testid="stVideo"] {
            max-height: 480px;
        }
        [data-testid="stImage"] img, [data-testid="stVideo"] video {
            max-height: 450px;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid #293545;
        }
        [data-testid="stAlert"] {
            border-radius: 8px;
            font-weight: 750;
        }
        .stButton > button {
            background: #202a37;
            border-color: #465568;
            color: #ffffff;
        }
        /* Compact cyber-industrial dashboard theme */
        :root {
            --bg: #0d0b18;
            --panel: #171329;
            --panel-strong: #1e1b2e;
            --border: #302747;
            --purple: #7c3aed;
            --purple-light: #9f7aea;
            --red: #ff4b4b;
            --green: #10b981;
            --text: #ffffff;
            --muted: #a99fbd;
        }
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background: var(--bg) !important;
            color: var(--text) !important;
        }
        body {
            overflow: hidden;
        }
        .block-container {
            max-width: 1560px;
            min-height: calc(100vh - 1rem);
            overflow: hidden;
            padding: 0.85rem 1.5rem 1rem;
        }
        [data-testid="stHeader"] {
            background: var(--bg) !important;
        }
        .dashboard-header {
            border-bottom: 1px solid var(--border);
            margin-bottom: 0.75rem;
            padding-bottom: 0.65rem;
        }
        .dashboard-header h1 {
            color: var(--text) !important;
            font-size: clamp(1.25rem, 2vw, 1.8rem);
        }
        .dashboard-header p {
            color: var(--muted) !important;
            font-size: 0.78rem;
            margin-top: 0.2rem;
        }
        .live-pill {
            background: rgba(16, 185, 129, 0.12);
            border-color: var(--green);
            color: #6ee7b7;
            font-size: 0.7rem;
            padding: 0.35rem 0.65rem;
        }
        .card-title {
            color: var(--purple-light) !important;
            font-size: 0.68rem;
            letter-spacing: 0.13em;
            margin-bottom: 0.45rem;
        }
        .profile-card, .dashboard-card {
            background: var(--panel) !important;
            border-color: var(--border) !important;
        }
        .profile-name {
            color: var(--text) !important;
            font-size: 1.125rem;
        }
        .profile-subtitle {
            color: var(--muted) !important;
            font-size: 0.7rem;
        }
        [data-testid="stImage"] {
            background: var(--panel-strong) !important;
            border-color: var(--purple) !important;
            border-radius: 10px;
            max-height: 380px;
            padding: 0.35rem;
        }
        [data-testid="stImage"] img {
            max-height: 360px;
            object-fit: contain;
        }
        [data-testid="stVideo"] {
            background: #08070f !important;
            border-color: var(--border) !important;
            max-height: 380px;
            padding: 0.35rem;
        }
        [data-testid="stVideo"] video {
            max-height: 360px;
            object-fit: contain;
        }
        [data-testid="stFileUploader"],
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"] {
            background: var(--panel-strong) !important;
            border-color: #584978 !important;
            color: var(--text) !important;
        }
        [data-testid="stFileUploader"] * {
            color: var(--text) !important;
        }
        [data-testid="stFileUploader"] small {
            color: var(--muted) !important;
        }
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"] {
            background: linear-gradient(135deg, #7c3aed, #5b21b6) !important;
            border: 1px solid #9f7aea !important;
            border-radius: 7px !important;
            box-shadow: 0 0 14px rgba(124, 58, 237, 0.28);
            color: #ffffff !important;
            font-size: 0.75rem !important;
            font-weight: 750 !important;
            min-height: 2.1rem;
            padding: 0.35rem 0.85rem !important;
        }
        [data-testid="stFileUploader"] button:hover,
        [data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"]:hover {
            background: linear-gradient(135deg, #8b5cf6, #6d28d9) !important;
            border-color: #c4b5fd !important;
            box-shadow: 0 0 18px rgba(159, 122, 234, 0.5);
            color: #ffffff !important;
        }
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] p,
        [data-testid="stSelectbox"] label,
        [data-testid="stExpander"] summary {
            color: var(--text) !important;
        }
        [data-testid="stRadio"] div[role="radiogroup"] {
            gap: 0.2rem;
        }
        [data-testid="stMetric"] {
            background: var(--panel) !important;
            border: 1px solid var(--border);
            border-left: 3px solid var(--purple);
            border-radius: 7px;
            padding: 0.45rem 0.55rem;
        }
        [data-testid="stMetricLabel"] {
            color: var(--muted) !important;
            font-size: 0.68rem;
        }
        [data-testid="stMetricValue"] {
            color: var(--text) !important;
            font-size: 1.25rem;
        }
        [data-testid="stDataFrame"] {
            max-height: 150px;
            overflow-y: auto;
        }
        [data-testid="stDataFrame"] *, [data-testid="stAlert"] * {
            color: var(--text) !important;
        }
        [data-testid="stAlert"] {
            background: var(--panel-strong) !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
            margin: 0.35rem 0;
            padding: 0.45rem 0.65rem;
        }
        .violation-banner {
            background: rgba(255, 75, 75, 0.14);
            border: 1px solid var(--red);
            border-radius: 7px;
            color: #ffd4d4;
            font-size: 0.8rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            margin: 0.35rem 0;
            padding: 0.55rem 0.7rem;
        }
        .status-light-card {
            align-items: center;
            background: var(--panel-strong);
            border: 1px solid var(--border);
            border-radius: 7px;
            display: flex;
            gap: 0.65rem;
            margin: 0.35rem 0 0.75rem;
            padding: 0.55rem 0.7rem;
        }
        .lamp-housing {
            align-items: center;
            background: #090b10;
            border: 2px solid #3b4555;
            border-radius: 50%;
            box-shadow: inset 0 0 6px #000, 0 2px 4px rgba(0, 0, 0, 0.35);
            display: flex;
            height: 5rem;
            justify-content: center;
            overflow: hidden;
            position: relative;
            width: 5rem;
        }
        .lamp-housing::before {
            animation: lamp-ring-spin 2.2s linear infinite;
            background: conic-gradient(
                transparent 0deg,
                transparent 235deg,
                rgba(255, 255, 255, 0.85) 275deg,
                transparent 315deg,
                transparent 360deg
            );
            border-radius: 50%;
            content: "";
            inset: -0.2rem;
            position: absolute;
        }
        .status-light {
            animation: lamp-pulse 1.15s ease-in-out infinite;
            border-radius: 50%;
            display: inline-block;
            height: 3.4rem;
            position: relative;
            width: 3.4rem;
            z-index: 1;
        }
        @keyframes lamp-pulse {
            0%, 100% {
                box-shadow: 0 0 14px var(--lamp-colour), 0 0 34px var(--lamp-colour), 0 0 64px var(--lamp-colour);
                transform: scale(0.9);
            }
            50% {
                box-shadow: 0 0 26px var(--lamp-colour), 0 0 62px var(--lamp-colour), 0 0 110px var(--lamp-colour);
                transform: scale(1.12);
            }
        }
        @keyframes lamp-ring-spin {
            to {
                transform: rotate(360deg);
            }
        }
        .status-light-card strong {
            color: var(--text);
            font-size: 0.78rem;
        }
        .status-light-card small {
            color: var(--muted);
            font-size: 0.65rem;
            margin-left: 0.35rem;
        }
        .stButton > button {
            background: var(--panel-strong) !important;
            border: 1px solid #584978 !important;
            color: var(--text) !important;
            font-size: 0.75rem;
        }
        .stButton > button:hover {
            border-color: var(--purple) !important;
            box-shadow: 0 0 12px rgba(124, 58, 237, 0.25);
        }
        [data-testid="stInfo"], [data-testid="stCaptionContainer"] {
            color: var(--muted) !important;
        }
        /* AI vision command-center skin */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Rajdhani:wght@500;600;700&display=swap');
        :root {
            --command-bg: #0b0f19;
            --command-panel: rgba(17, 27, 43, 0.78);
            --command-panel-strong: #111827;
            --command-border: rgba(0, 229, 255, 0.2);
            --command-cyan: #00e5ff;
            --command-amber: #ffb800;
            --command-green: #10b981;
            --command-red: #ef4444;
        }
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background:
                radial-gradient(circle at 12% 4%, rgba(0, 229, 255, 0.08), transparent 25rem),
                radial-gradient(circle at 92% 18%, rgba(255, 184, 0, 0.06), transparent 24rem),
                linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px),
                var(--command-bg) !important;
            background-size: auto, auto, 34px 34px, 34px 34px, auto !important;
            font-family: Inter, sans-serif !important;
        }
        body {
            overflow: auto;
        }
        .block-container {
            max-width: 1640px;
            padding: 1.1rem 1.6rem 2.5rem;
        }
        .dashboard-header {
            background: linear-gradient(110deg, rgba(17, 24, 39, 0.94), rgba(15, 23, 42, 0.62));
            border: 1px solid var(--command-border);
            border-left: 3px solid var(--command-amber);
            border-radius: 12px;
            box-shadow: 0 0 32px rgba(0, 229, 255, 0.06);
            margin-bottom: 1rem;
            padding: 0.8rem 1rem;
        }
        .dashboard-header p {
            color: #b6c5d8 !important;
            font-family: Rajdhani, Inter, sans-serif;
            font-size: 0.86rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .live-pill {
            background: rgba(16, 185, 129, 0.1) !important;
            border: 1px solid var(--command-green) !important;
            box-shadow: 0 0 16px rgba(16, 185, 129, 0.16);
            color: #6ee7b7 !important;
            font-family: Rajdhani, Inter, sans-serif;
            letter-spacing: 0.08em;
        }
        .card-title {
            color: var(--command-amber) !important;
            font-family: Rajdhani, Inter, sans-serif;
            font-size: 0.76rem !important;
            letter-spacing: 0.18em !important;
            text-shadow: 0 0 10px rgba(255, 184, 0, 0.2);
        }
        .profile-name {
            font-family: Rajdhani, Inter, sans-serif;
            letter-spacing: 0.1em;
        }
        .profile-subtitle {
            color: #8ea4bd !important;
        }
        .operator-photo-hud {
            display: inline-block;
            position: relative;
        }
        .operator-photo-hud::before,
        .operator-photo-hud::after {
            border-color: var(--command-cyan);
            border-style: solid;
            content: "";
            height: 16px;
            position: absolute;
            width: 16px;
            z-index: 3;
        }
        .operator-photo-hud::before {
            border-width: 2px 0 0 2px;
            left: -4px;
            top: -4px;
        }
        .operator-photo-hud::after {
            border-width: 0 2px 2px 0;
            bottom: -4px;
            right: -4px;
        }
        .status-section-title,
        .panel-section-title {
            margin-top: 1rem !important;
        }
        .operator-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.3rem;
            justify-content: center;
            margin-top: 0.7rem;
        }
        .operator-badges span {
            background: rgba(0, 229, 255, 0.07);
            border: 1px solid rgba(0, 229, 255, 0.2);
            border-radius: 999px;
            color: #9bd8e0;
            font-family: Rajdhani, Inter, sans-serif;
            font-size: 0.59rem;
            letter-spacing: 0.05em;
            padding: 0.22rem 0.42rem;
            transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
        }
        .operator-badges span::first-letter {
            color: var(--command-cyan);
        }
        .operator-badges span:hover {
            border-color: var(--command-cyan);
            box-shadow: 0 0 12px rgba(0, 229, 255, 0.28);
            transform: translateY(-1px);
        }
        .feed-heading {
            align-items: center;
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.35rem;
        }
        .feed-heading .card-title {
            margin-bottom: 0 !important;
        }
        .feed-badge {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.65);
            border-radius: 999px;
            box-shadow: 0 0 14px rgba(239, 68, 68, 0.2);
            color: #fca5a5;
            font-family: Rajdhani, Inter, sans-serif;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            padding: 0.25rem 0.55rem;
        }
        [data-testid="stImage"], [data-testid="stVideo"] {
            background: #070b12 !important;
            border: 1px solid rgba(0, 229, 255, 0.32) !important;
            border-radius: 14px !important;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.025), 0 0 26px rgba(0, 229, 255, 0.08);
            padding: 0.55rem !important;
            position: relative;
        }
        [data-testid="stImage"]::before, [data-testid="stVideo"]::before {
            border-left: 2px solid var(--command-cyan);
            border-top: 2px solid var(--command-cyan);
            content: "";
            height: 22px;
            left: 5px;
            position: absolute;
            top: 5px;
            width: 22px;
            z-index: 2;
        }
        [data-testid="stImage"]::after, [data-testid="stVideo"]::after {
            border-bottom: 2px solid var(--command-cyan);
            border-right: 2px solid var(--command-cyan);
            bottom: 5px;
            content: "";
            height: 22px;
            position: absolute;
            right: 5px;
            width: 22px;
            z-index: 2;
        }
        [data-testid="stImage"] img, [data-testid="stVideo"] video {
            filter: saturate(1.05) contrast(1.03);
        }
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(23, 37, 58, 0.9), rgba(12, 18, 30, 0.9)) !important;
            border: 1px solid rgba(148, 163, 184, 0.16) !important;
            border-left: 3px solid var(--command-amber) !important;
            border-radius: 10px !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 8px 24px rgba(0,0,0,0.2);
            min-height: 76px;
        }
        [data-testid="stMetric"]:nth-child(2) {
            border-left-color: var(--command-green) !important;
        }
        [data-testid="stMetric"]:nth-child(3) {
            border-left-color: var(--command-red) !important;
        }
        [data-testid="stMetricLabel"] {
            color: #8ea4bd !important;
            font-family: Rajdhani, Inter, sans-serif;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        [data-testid="stMetricValue"] {
            color: #f8fafc !important;
            font-family: Rajdhani, Inter, sans-serif;
            font-size: 2rem !important;
            text-shadow: 0 0 12px rgba(0, 229, 255, 0.18);
        }
        [data-testid="stRadio"] [role="radiogroup"] {
            background: rgba(17, 24, 39, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 10px;
            padding: 0.55rem 0.7rem;
        }
        [data-testid="stFileUploader"],
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(17, 24, 39, 0.74) !important;
            border: 1px dashed rgba(0, 229, 255, 0.34) !important;
            border-radius: 10px !important;
        }
        .stButton > button {
            background: linear-gradient(135deg, #ffb800, #d97706) !important;
            border: 1px solid #ffd166 !important;
            border-radius: 8px !important;
            box-shadow: 0 0 16px rgba(255, 184, 0, 0.16);
            color: #111827 !important;
            font-family: Rajdhani, Inter, sans-serif;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .stButton > button:hover {
            box-shadow: 0 0 24px rgba(255, 184, 0, 0.42);
            transform: translateY(-1px);
        }
        .status-light-card {
            background: linear-gradient(145deg, rgba(25, 35, 54, 0.9), rgba(12, 18, 30, 0.85)) !important;
            border-color: rgba(0, 229, 255, 0.22) !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 0 22px rgba(0, 229, 255, 0.08);
            min-height: 5rem;
        }
        .terminal-heading {
            border-bottom: 1px solid rgba(0, 229, 255, 0.24);
            color: var(--command-cyan);
            font-family: Rajdhani, Inter, sans-serif;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            margin-top: 1rem;
            padding: 0.55rem 0;
        }
        .terminal-dot {
            background: var(--command-cyan);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--command-cyan);
            display: inline-block;
            height: 7px;
            margin-right: 0.45rem;
            width: 7px;
        }
        .terminal-state {
            color: #6ee7b7;
            float: right;
            font-size: 0.65rem;
        }
        [data-testid="stDataFrame"] {
            background: rgba(5, 10, 18, 0.72);
            border: 1px solid rgba(0, 229, 255, 0.2) !important;
            border-radius: 8px;
            font-family: "Cascadia Code", Consolas, monospace;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    try:
        model = load_model()
    except (FileNotFoundError, OSError, RuntimeError) as error:
        st.error(f"Detection engine unavailable: {error}")
        st.stop()
    source = "Image upload"
    uploaded_image = None
    uploaded_video = None
    left_column, right_column = st.columns([2.15, 1], gap="large")
    with right_column:
        st.markdown('<div class="card-title">Site safety operator</div>', unsafe_allow_html=True)
        st.markdown('<div class="operator-photo-hud">', unsafe_allow_html=True)
        if PROFILE_IMAGE_PATH.exists():
            st.image(str(PROFILE_IMAGE_PATH), width=110)
        else:
            st.warning(f"Profile image not found: {PROFILE_IMAGE_PATH.name}")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="profile-name">ALKHALIL AHMED</div>'
            '<div class="profile-subtitle">Computer Vision &amp; AI Safety Specialist</div>'
            '<div class="operator-badges">'
            '<span>🪖 HARD HAT</span><span>🦺 SAFETY VEST</span><span>🛡️ AI SHIELD</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        source = st.session_state.get("inspection_source", "Image upload")
        st.markdown('<div class="card-title status-section-title">Compliance status</div>', unsafe_allow_html=True)
        status_placeholder = st.empty()
        if source == "Live camera":
            render_live_status_light()
        else:
            render_status_light(get_live_status(), status_placeholder)
        st.markdown('<div class="card-title panel-section-title">Input source</div>', unsafe_allow_html=True)
        source = st.radio(
            "Choose inspection source",
            ["Image upload", "Video upload", "Live camera"],
            key="inspection_source",
            label_visibility="collapsed",
        )
        if source == "Image upload":
            uploaded_image = st.file_uploader(
                "Select an inspection image",
                type=["jpg", "jpeg", "png"],
                key="image",
            )
        elif source == "Video upload":
            uploaded_video = st.file_uploader(
                "Select a site video",
                type=["mp4", "avi", "mov", "mkv"],
                key="video",
            )
        st.markdown('<div class="card-title" style="margin-top:1.4rem;">Quick stats</div>', unsafe_allow_html=True)
        render_kpis()
        st.caption(f"Model: {MODEL_PATH.name}")
        with st.expander("Loaded model classes"):
            st.write(model.names)
        if st.button("Clear violation log", type="secondary", use_container_width=True):
            st.session_state.violation_log.clear()
            LIVE_VIOLATION_LOG.clear()
            reset_live_status()
            st.rerun()

    with left_column:
        st.markdown(
            '<div class="feed-heading"><span class="card-title">Live inspection feed</span>'
            '<span class="feed-badge">● LIVE</span></div>',
            unsafe_allow_html=True,
        )
        if source == "Image upload":
            if uploaded_image:
                process_image(model, uploaded_image, status_placeholder)
            else:
                st.info("Upload an image to start the PPE inspection.")
        elif source == "Video upload":
            if uploaded_video:
                process_video(model, uploaded_video, status_placeholder)
            else:
                st.info("Upload a video to start frame-by-frame PPE inspection.")
        else:
            render_live_camera(model)
        st.markdown('<div class="card-title" style="margin-top:1.3rem;">Real-time violation log</div>', unsafe_allow_html=True)
        render_log()


if __name__ == "__main__":
    main()
