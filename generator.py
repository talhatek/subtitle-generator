#!/usr/bin/env python3
"""
SRT Subtitle Generator from Video/Audio files.
Uses faster-whisper (CTranslate2) for speech-to-text with GPU acceleration.

Usage:
    python generator.py video.mp4
    python generator.py video.mp4 --model large-v3 --language de
    python generator.py audio.mp3 --output subtitles.srt
"""

import argparse
import sys
import time
from pathlib import Path

import av
import numpy as np
from faster_whisper import WhisperModel

# Folders (same convention as main.py)
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".flv", ".wmv",
                    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"}

# Subtitle formatting constants
MAX_CHARS_PER_LINE = 42
MAX_LINES = 2
MIN_SUBTITLE_DURATION = 0.5  # seconds
MAX_SUBTITLE_DURATION = 7.0  # seconds

# Waveform gap detection constants
GAP_THRESHOLD_RATIO = 0.15   # silence = 15% of mean RMS in search window
GAP_PADDING_MS = 30          # padding (ms) between speech edge and subtitle boundary
MIN_GAP_MS = 50              # minimum gap (ms) to enforce between consecutive subtitles
MERGE_GAP_MS = 300           # merge bursts separated by less than this (inter-word silence)


def format_timestamp(seconds):
    """Convert seconds (float) to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    # Clamp millis to 999 (rounding can push to 1000)
    if millis >= 1000:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def split_text_into_lines(text):
    """
    Split subtitle text into at most MAX_LINES lines,
    each at most MAX_CHARS_PER_LINE characters.

    Splits at word boundaries near the midpoint for balanced lines.
    Returns the formatted text with newlines.
    """
    text = text.strip()

    # If it fits on one line, done
    if len(text) <= MAX_CHARS_PER_LINE:
        return text

    words = text.split()
    total_len = len(text)

    # Try to split into 2 balanced lines
    # Find the split point closest to the midpoint
    target = total_len / 2
    best_split = 0
    best_diff = total_len
    current_len = 0

    for i, word in enumerate(words[:-1]):  # Don't split after last word
        current_len += len(word) + (1 if i > 0 else 0)
        diff = abs(current_len - target)
        if diff < best_diff:
            best_diff = diff
            best_split = i + 1

    if best_split == 0:
        best_split = 1

    line1 = " ".join(words[:best_split])
    line2 = " ".join(words[best_split:])

    # Check if both lines fit within the char limit
    if len(line1) <= MAX_CHARS_PER_LINE and len(line2) <= MAX_CHARS_PER_LINE:
        return f"{line1}\n{line2}"

    # If line1 is too long, try splitting at MAX_CHARS_PER_LINE boundary
    if len(line1) > MAX_CHARS_PER_LINE:
        # Find the last word that fits in MAX_CHARS_PER_LINE
        line1 = ""
        split_idx = 0
        for i, word in enumerate(words):
            test = f"{line1} {word}".strip()
            if len(test) > MAX_CHARS_PER_LINE:
                break
            line1 = test
            split_idx = i + 1
        if split_idx == 0:
            split_idx = 1
            line1 = words[0]
        line2 = " ".join(words[split_idx:])

    # Truncate line2 if still too long (hard limit)
    if len(line2) > MAX_CHARS_PER_LINE:
        truncated_words = []
        current = ""
        for word in line2.split():
            test = f"{current} {word}".strip()
            if len(test) > MAX_CHARS_PER_LINE:
                break
            current = test
            truncated_words.append(word)
        line2 = " ".join(truncated_words) if truncated_words else line2[:MAX_CHARS_PER_LINE]

    return f"{line1}\n{line2}"


def segment_needs_splitting(segment):
    """
    Check if a segment's text is too long for a single subtitle entry
    (exceeds 2 lines of MAX_CHARS_PER_LINE).
    """
    text = segment.text.strip()
    if len(text) <= MAX_CHARS_PER_LINE * MAX_LINES + 1:  # +1 for newline
        return False
    # More precise check: can split_text_into_lines handle it without loss?
    words = text.split()
    total_chars_if_split = sum(len(w) for w in words) + len(words) - 1
    return total_chars_if_split > MAX_CHARS_PER_LINE * MAX_LINES


def split_segment_by_words(segment):
    """
    Split a long segment into multiple subtitle entries using word-level timestamps.
    Returns a list of (start, end, text) tuples.
    """
    if not segment.words:
        # No word timestamps available, just use the segment as-is
        return [(segment.start, segment.end, segment.text.strip())]

    entries = []
    current_words = []
    current_text = ""
    current_start = segment.words[0].start

    for word in segment.words:
        test_text = f"{current_text} {word.word}".strip()

        # Check if adding this word would exceed 2 lines
        formatted = split_text_into_lines(test_text)
        line_count = formatted.count("\n") + 1
        max_line_len = max(len(line) for line in formatted.split("\n"))

        if line_count > MAX_LINES or max_line_len > MAX_CHARS_PER_LINE:
            # Flush current entry
            if current_words:
                entries.append((
                    current_start,
                    current_words[-1].end,
                    current_text.strip(),
                ))
            # Start new entry with this word
            current_words = [word]
            current_text = word.word.strip()
            current_start = word.start
        else:
            current_words.append(word)
            current_text = test_text

    # Don't forget the last entry
    if current_words:
        entries.append((
            current_start,
            current_words[-1].end,
            current_text.strip(),
        ))

    return entries


def build_subtitle_entries(segments):
    """
    Process raw transcription segments into formatted subtitle entries.

    Returns a list of dicts with: index, start, end, text
    """
    entries = []
    index = 1

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        # Check if the segment needs to be split into multiple entries
        if segment_needs_splitting(segment) and segment.words:
            sub_entries = split_segment_by_words(segment)
            for start, end, sub_text in sub_entries:
                if not sub_text:
                    continue
                # Ensure minimum duration
                if end - start < MIN_SUBTITLE_DURATION:
                    end = start + MIN_SUBTITLE_DURATION
                entries.append({
                    "index": index,
                    "start": start,
                    "end": end,
                    "text": split_text_into_lines(sub_text),
                })
                index += 1
        else:
            start = segment.start
            end = segment.end
            # Ensure minimum duration
            if end - start < MIN_SUBTITLE_DURATION:
                end = start + MIN_SUBTITLE_DURATION
            # Cap maximum duration
            if end - start > MAX_SUBTITLE_DURATION and segment.words:
                # If too long and we have word timestamps, split it
                sub_entries = split_segment_by_words(segment)
                for s, e, sub_text in sub_entries:
                    if not sub_text:
                        continue
                    if e - s < MIN_SUBTITLE_DURATION:
                        e = s + MIN_SUBTITLE_DURATION
                    entries.append({
                        "index": index,
                        "start": s,
                        "end": e,
                        "text": split_text_into_lines(sub_text),
                    })
                    index += 1
            else:
                entries.append({
                    "index": index,
                    "start": start,
                    "end": end,
                    "text": split_text_into_lines(text),
                })
                index += 1

    return entries


def extract_audio(video_path):
    """
    Extract mono audio from a video/audio file using PyAV.
    Returns (samples, sample_rate) where samples is a float32 numpy array
    normalized to [-1.0, 1.0].

    Skips corrupt packets gracefully (same approach as faster-whisper).
    """
    container = av.open(str(video_path), metadata_errors="ignore")
    audio_stream = container.streams.audio[0]
    sample_rate = audio_stream.rate or audio_stream.codec_context.sample_rate

    # Use AudioResampler to convert any format/layout to mono float32
    resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)

    frames = []
    decoder = container.decode(audio=0)
    while True:
        try:
            frame = next(decoder)
        except StopIteration:
            break
        except av.error.InvalidDataError:
            continue  # skip corrupt packets, keep decoding

        # Resample to mono float32
        resampled = resampler.resample(frame)
        for rf in resampled:
            arr = rf.to_ndarray().flatten()
            frames.append(arr)

    # Flush the resampler (get any remaining buffered samples)
    resampled = resampler.resample(None)
    for rf in resampled:
        arr = rf.to_ndarray().flatten()
        frames.append(arr)

    container.close()

    if not frames:
        return np.array([], dtype=np.float32), sample_rate

    samples = np.concatenate(frames)
    return samples, sample_rate


def compute_rms(samples, sr, start_sec, end_sec, frame_ms=10):
    """
    Compute RMS energy in non-overlapping frames for the audio between
    start_sec and end_sec.

    Returns a 1-D numpy array of RMS values, one per frame.
    """
    start_idx = max(0, int(start_sec * sr))
    end_idx = min(len(samples), int(end_sec * sr))
    segment = samples[start_idx:end_idx]

    frame_size = int(sr * frame_ms / 1000)
    if frame_size == 0 or len(segment) == 0:
        return np.array([])

    # Trim to whole frames
    n_frames = len(segment) // frame_size
    if n_frames == 0:
        return np.array([np.sqrt(np.mean(segment ** 2))])

    segment = segment[:n_frames * frame_size]
    frames = segment.reshape(n_frames, frame_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    return rms


def find_speech_bursts(samples, sr, region_start, region_end, silence_threshold,
                       min_burst_ms=100):
    """
    Find all continuous speech bursts within a time region.

    A "burst" is a contiguous run of frames where RMS exceeds the silence
    threshold. Short bursts (< min_burst_ms) are discarded as noise/clicks.

    Returns a list of (burst_start, burst_end) tuples in seconds (with padding),
    sorted by start time. Returns empty list if no bursts found.
    """
    rms = compute_rms(samples, sr, region_start, region_end, frame_ms=10)
    if len(rms) == 0:
        return []

    is_loud = rms >= silence_threshold
    frame_duration = 0.010
    min_burst_frames = max(1, int(min_burst_ms / 10))
    padding_sec = GAP_PADDING_MS / 1000.0

    bursts = []
    burst_start_frame = None

    for i in range(len(rms)):
        if is_loud[i]:
            if burst_start_frame is None:
                burst_start_frame = i
        else:
            if burst_start_frame is not None:
                burst_len = i - burst_start_frame
                if burst_len >= min_burst_frames:
                    bs = region_start + burst_start_frame * frame_duration - padding_sec
                    be = region_start + i * frame_duration + padding_sec
                    bursts.append((max(region_start, bs), min(region_end, be)))
                burst_start_frame = None

    # Handle burst that extends to end of region
    if burst_start_frame is not None:
        burst_len = len(rms) - burst_start_frame
        if burst_len >= min_burst_frames:
            bs = region_start + burst_start_frame * frame_duration - padding_sec
            be = region_start + len(rms) * frame_duration + padding_sec
            bursts.append((max(region_start, bs), min(region_end, be)))

    return bursts


def find_longest_burst(bursts):
    """Return the longest burst from a list of (start, end) tuples, or None."""
    if not bursts:
        return None
    return max(bursts, key=lambda b: b[1] - b[0])


def merge_nearby_bursts(bursts, max_gap_ms=None):
    """
    Merge bursts that are separated by less than max_gap_ms milliseconds.

    Inter-word silences in normal speech are typically 100-250ms. Bursts
    separated by less than the merge threshold are part of the same utterance
    and should be treated as one continuous speech region.

    Returns a new list of merged (start, end) tuples.
    """
    if not bursts:
        return []
    if max_gap_ms is None:
        max_gap_ms = MERGE_GAP_MS

    max_gap_sec = max_gap_ms / 1000.0
    merged = [list(bursts[0])]

    for bs, be in bursts[1:]:
        prev = merged[-1]
        if bs - prev[1] <= max_gap_sec:
            # Close enough — extend the previous burst
            prev[1] = max(prev[1], be)
        else:
            merged.append([bs, be])

    return [(s, e) for s, e in merged]


def post_process_gaps(entries, samples, sr):
    """
    Refine ALL subtitle timestamps using waveform analysis.

    For each subtitle, scans its entire Whisper-assigned timespan (plus
    neighboring gaps) to find where speech actually lives. Uses speech burst
    detection to identify the longest continuous speech region, handling:
    - Butting timestamps (gap ~0ms) where silence exists but isn't reflected
    - Overly large gaps where Whisper started a subtitle seconds too early
    - Whisper including long silence within a subtitle's timespan
    - Short noise bursts that don't belong to the subtitle's text

    Uses a global silence threshold computed from the full audio track.

    Modifies entries in-place and returns the number of timestamps adjusted.
    """
    if len(entries) < 1:
        return 0

    adjusted = 0
    min_gap_sec = MIN_GAP_MS / 1000.0
    audio_duration = len(samples) / sr

    # Compute global silence threshold from the full audio track
    full_rms = compute_rms(samples, sr, 0.0, audio_duration, frame_ms=10)
    if len(full_rms) == 0:
        return 0
    mean_rms = np.mean(full_rms)
    if mean_rms < 1e-8:
        return 0
    silence_threshold = mean_rms * GAP_THRESHOLD_RATIO
    print(f"  Global RMS: {mean_rms:.4f}, silence threshold: {silence_threshold:.4f}")

    # --- Pass 1: For each subtitle, find the longest speech burst in its region ---
    for i, entry in enumerate(entries):
        # Define the search region: the subtitle's own span, extended into
        # neighboring gaps to catch speech that Whisper misaligned.
        # We search from the midpoint of the gap before, to the midpoint of
        # the gap after. This ensures each subtitle "owns" half of each gap.
        if i > 0:
            prev_end = entries[i - 1]["end"]
            search_start = (prev_end + entry["start"]) / 2.0
        else:
            search_start = max(0.0, entry["start"] - 0.5)

        if i < len(entries) - 1:
            next_start = entries[i + 1]["start"]
            search_end = (entry["end"] + next_start) / 2.0
        else:
            search_end = min(audio_duration, entry["end"] + 0.5)

        # Make sure search region covers at least the subtitle's own range
        search_start = min(search_start, entry["start"])
        search_end = max(search_end, entry["end"])

        bursts = find_speech_bursts(samples, sr, search_start, search_end,
                                    silence_threshold)
        if not bursts:
            continue

        # Merge nearby bursts (inter-word silences) into utterance-level regions
        bursts = merge_nearby_bursts(bursts)

        # Use the longest burst — this is the main speech for this subtitle
        best = find_longest_burst(bursts)
        if best is None:
            continue

        new_start, new_end = best

        # Safety: subtitle must be at least MIN_SUBTITLE_DURATION long
        if new_end - new_start < MIN_SUBTITLE_DURATION:
            continue

        # Track changes
        if abs(new_start - entry["start"]) > 0.005:
            adjusted += 1
        if abs(new_end - entry["end"]) > 0.005:
            adjusted += 1

        entry["start"] = new_start
        entry["end"] = new_end

    # --- Pass 2: Fix any overlaps or too-small gaps ---
    for i in range(len(entries) - 1):
        current = entries[i]
        nxt = entries[i + 1]

        gap = nxt["start"] - current["end"]

        if gap < min_gap_sec:
            # Overlapping or gap too small — split the difference
            midpoint = (current["end"] + nxt["start"]) / 2.0
            half_gap = min_gap_sec / 2.0
            new_end = midpoint - half_gap
            new_start = midpoint + half_gap

            # Make sure we don't make subtitles too short
            if (new_end - current["start"] >= MIN_SUBTITLE_DURATION and
                    nxt["end"] - new_start >= MIN_SUBTITLE_DURATION):
                current["end"] = new_end
                nxt["start"] = new_start

    return adjusted


def write_srt(entries, output_path):
    """Write subtitle entries to an SRT file."""
    blocks = []
    for entry in entries:
        start_ts = format_timestamp(entry["start"])
        end_ts = format_timestamp(entry["end"])
        block = f"{entry['index']}\n{start_ts} --> {end_ts}\n{entry['text']}"
        blocks.append(block)

    content = "\n\n".join(blocks) + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return len(entries)


def load_model(model_size, device):
    """
    Load the faster-whisper model with appropriate device and compute type.
    Falls back to CPU if CUDA fails.
    """
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"

    if device == "cuda":
        compute_type = "float16"
    else:
        compute_type = "int8"

    print(f"Loading model: {model_size}")
    print(f"Device: {device} ({compute_type})")

    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        if device == "cuda":
            print(f"CUDA load failed: {e}")
            print("Falling back to CPU (int8)...")
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            print("Model loaded on CPU.")
            return model
        raise


def transcribe_audio(model, input_path, language):
    """
    Transcribe audio/video file using faster-whisper.
    Returns a list of segment objects.
    """
    print(f"\nTranscribing: {input_path}")
    print(f"Language: {language or 'auto-detect'}")
    print("This may take a while...\n")

    start_time = time.time()

    segments_gen, info = model.transcribe(
        str(input_path),
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=300,
            speech_pad_ms=200,
        ),
    )

    if info.language:
        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    # Consume the generator to get all segments (with progress feedback)
    segments = []
    for segment in segments_gen:
        segments.append(segment)
        # Print progress: show each segment as it's transcribed
        print(
            f"  [{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}] "
            f"{segment.text.strip()}"
        )

    elapsed = time.time() - start_time
    print(f"\nTranscription completed in {elapsed:.1f}s")
    print(f"Segments found: {len(segments)}")

    return segments


def find_media_files(folder):
    """Find all video/audio files in the given folder."""
    folder = Path(folder)
    if not folder.exists():
        return []
    files = []
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
            files.append(f)
    return files


def process_file(model, input_path, output_path, language):
    """Process a single video/audio file: transcribe and generate SRT."""
    print(f"\n{'=' * 60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"{'=' * 60}")

    try:
        # Transcribe
        segments = transcribe_audio(model, input_path, language)

        if not segments:
            print("No speech detected in the input file.")
            return False

        # Format subtitles with smart splitting
        print("\nFormatting subtitles...")
        entries = build_subtitle_entries(segments)
        print(f"Subtitle entries: {len(entries)}")

        # Post-process: refine gaps using waveform analysis
        try:
            print("Extracting audio for gap detection...")
            samples, sr = extract_audio(input_path)
            print(f"Audio: {len(samples)} samples at {sr} Hz ({len(samples)/sr:.1f}s)")
            adjusted = post_process_gaps(entries, samples, sr)
            print(f"Gap detection: adjusted {adjusted} subtitle boundaries")
        except Exception as e:
            print(f"Gap detection skipped (non-fatal): {e}")

        # Write SRT file
        count = write_srt(entries, output_path)

        print(f"\nDone! Generated {count} subtitle entries.")
        print(f"Output saved to: {output_path}")
        return True

    except Exception as e:
        print(f"Failed to process {input_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate SRT subtitles from video/audio files using faster-whisper.",
        epilog=(
            "Examples:\n"
            "  python generator.py                          # process all files in input/\n"
            "  python generator.py video.mp4                # process a specific file\n"
            "  python generator.py --model medium --language en\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help="Path to input video/audio file. If omitted, processes all "
             "files in the input/ folder (default behavior).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3", "turbo",
                 "distil-large-v3", "distil-medium.en"],
        help="Whisper model size (default: large-v3)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="de",
        help="Language code, e.g. 'de' for German, 'en' for English (default: de). "
             "Use 'auto' for auto-detection.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .srt file path (only when processing a single file). "
             "Default: output/<input_stem>.srt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto)",
    )

    args = parser.parse_args()

    # Handle language
    language = args.language if args.language != "auto" else None

    # Build list of (input_path, output_path) pairs
    jobs = []

    if args.input:
        # Explicit file provided
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)
        if args.output:
            output_path = Path(args.output)
        else:
            Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
            output_path = Path(OUTPUT_FOLDER) / (input_path.stem + ".srt")
        jobs.append((input_path, output_path))
    else:
        # Auto-scan input/ folder
        Path(INPUT_FOLDER).mkdir(exist_ok=True)
        Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
        media_files = find_media_files(INPUT_FOLDER)
        if not media_files:
            print(f"No video/audio files found in '{INPUT_FOLDER}/' folder.")
            print(f"Place your .mp4, .mkv, .mp3, etc. files in the '{INPUT_FOLDER}/' folder,")
            print("or specify a file directly: python generator.py path/to/video.mp4")
            sys.exit(1)
        for f in media_files:
            output_path = Path(OUTPUT_FOLDER) / (f.stem + ".srt")
            jobs.append((f, output_path))

    print("=" * 60)
    print("SRT Subtitle Generator")
    print("=" * 60)
    print(f"Files:    {len(jobs)}")
    print(f"Model:    {args.model}")
    print(f"Language: {language or 'auto-detect'}")
    print(f"Device:   {args.device}")
    print("=" * 60)

    # Load model once (reuse for all files)
    model = load_model(args.model, args.device)

    # Process each file
    successful = 0
    failed = 0

    for input_path, output_path in jobs:
        if process_file(model, input_path, output_path, language):
            successful += 1
        else:
            failed += 1

    # Summary
    if len(jobs) > 1:
        print(f"\n{'=' * 60}")
        print("All files processed!")
        print(f"  Successful: {successful}")
        print(f"  Failed:     {failed}")
        print(f"  Output:     {OUTPUT_FOLDER}/")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
