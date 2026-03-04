#!/usr/bin/env python3
"""
SRT Batch Translator (English -> German)
Optimized for language learning - preserves slang, profanity, tech terms

Uses text-only batched translation to minimize token usage:
- Parses SRT into structured entries
- Sends only text + index to the model (no timestamps)
- Reassembles SRT with original timestamps + translated text
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load API key from .env file
load_dotenv()

# Configuration
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
MODEL = "openai/gpt-oss-120b"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
BATCH_SIZE = 30  # Number of subtitle entries per API call (tune based on TPM limit)

# System prompt - optimized for text-only JSON translation
SYSTEM_PROMPT = """You are a professional subtitle translator.
Task: Translate subtitle texts from English to German.
Objective: The user is learning German.

You will receive a JSON array of objects with "index" and "text" fields.
Return a JSON array with the same structure, where each "text" is translated to German.

Rules:
1. Return ONLY a valid JSON array. No markdown, no explanations, no extra text.
2. Keep the exact same "index" values. Do NOT add or remove entries.
3. Translate faithfully, including profanity, slang, and crude humor (do NOT sanitize).
4. Keep proper nouns (names, companies like 'Hooli', 'Pied Piper') in English.
5. Keep tech terms in English if commonly used in German tech (e.g., 'Code', 'App', 'Startup').
6. Ensure the German sounds natural and spoken, not robotic.
7. Preserve line breaks within text (\\n) exactly as they appear.

Example input:
[{"index": 1, "text": "Holy shit!"}, {"index": 2, "text": "What the fuck is that?"}]

Example output:
[{"index": 1, "text": "Heilige Scheiße!"}, {"index": 2, "text": "Was zur Hölle ist das?"}]"""


def parse_srt_entries(content):
    """
    Parse raw SRT content into a list of structured entries.

    Each entry is a dict with:
      - index: int (sequence number)
      - start: str (start timestamp, e.g. "00:00:13,360")
      - end: str (end timestamp)
      - text: str (subtitle text, may contain newlines)

    Returns list of entry dicts.
    """
    # Remove BOM if present
    content = content.lstrip("\ufeff")

    entries = []
    # Split on blank lines to get individual subtitle blocks
    blocks = re.split(r"\n\s*\n", content.strip())

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        lines = block.split("\n")
        if len(lines) < 3:
            continue

        # Line 1: sequence number
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        # Line 2: timestamp line "HH:MM:SS,mmm --> HH:MM:SS,mmm"
        timestamp_match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            lines[1].strip(),
        )
        if not timestamp_match:
            continue

        start = timestamp_match.group(1)
        end = timestamp_match.group(2)

        # Lines 3+: subtitle text (may be multi-line)
        text = "\n".join(lines[2:])

        entries.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "text": text,
            }
        )

    return entries


def build_srt_from_entries(entries):
    """
    Reconstruct a valid SRT file string from structured entries.

    Each entry must have: index, start, end, text
    """
    blocks = []
    for entry in entries:
        block = (
            f"{entry['index']}\n{entry['start']} --> {entry['end']}\n{entry['text']}"
        )
        blocks.append(block)
    return "\n\n".join(blocks) + "\n"


def translate_batch(client, batch, filename, batch_num, total_batches):
    """
    Translate a single batch of subtitle entries.

    Args:
        client: OpenAI client
        batch: list of {"index": int, "text": str} dicts
        filename: for logging
        batch_num: current batch number (1-based)
        total_batches: total number of batches

    Returns:
        list of {"index": int, "text": str} dicts with translated text
    """
    # Build the payload - only index and text, no timestamps
    payload = [{"index": entry["index"], "text": entry["text"]} for entry in batch]
    payload_json = json.dumps(payload, ensure_ascii=False)

    for attempt in range(MAX_RETRIES):
        raw_response = ""
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": payload_json},
                ],
                temperature=0.3,
                max_tokens=4096,
            )

            raw_response = response.choices[0].message.content.strip()

            # Clean up response - remove markdown code fences if model adds them
            if raw_response.startswith("```"):
                raw_response = re.sub(r"^```(?:json)?\s*", "", raw_response)
                raw_response = re.sub(r"\s*```$", "", raw_response)

            translated = json.loads(raw_response)

            # Validate: same number of entries
            if len(translated) != len(batch):
                print(
                    f"\n  Warning: {filename} batch {batch_num}/{total_batches} - "
                    f"count mismatch (expected {len(batch)}, got {len(translated)}). Retrying..."
                )
                time.sleep(RETRY_DELAY)
                continue

            # Validate: all indices present
            expected_indices = {entry["index"] for entry in batch}
            received_indices = {entry["index"] for entry in translated}
            if expected_indices != received_indices:
                print(
                    f"\n  Warning: {filename} batch {batch_num}/{total_batches} - "
                    f"index mismatch. Retrying..."
                )
                time.sleep(RETRY_DELAY)
                continue

            return translated

        except json.JSONDecodeError as e:
            print(
                f"\n  Warning: {filename} batch {batch_num}/{total_batches} - "
                f"invalid JSON response: {e}"
            )
            if attempt < MAX_RETRIES - 1:
                print(f"  Retrying in {RETRY_DELAY}s... ({attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  Failed after {MAX_RETRIES} attempts. Raw response:")
                print(f"  {raw_response[:500]}")
                raise

        except Exception as e:
            print(
                f"\n  Error translating {filename} batch {batch_num}/{total_batches}: {e}"
            )
            if attempt < MAX_RETRIES - 1:
                print(f"  Retrying in {RETRY_DELAY}s... ({attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                raise

    return None


def process_srt_file(client, index, pbar, weight, input_path, output_path):
    """Process a single SRT file using batched text-only translation."""
    filename = input_path.name

    try:
        # Read and parse the SRT file
        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read()

        entries = parse_srt_entries(content)
        if not entries:
            print(f"\n  Failed: Could not parse any subtitle entries from {filename}")
            return False

    
        # Split entries into batches
        batches = [
            entries[i : i + BATCH_SIZE] for i in range(0, len(entries), BATCH_SIZE)
        ]
        total_batches = len(batches)

        batch_weight = weight / total_batches
        # Translate each batch
        translated_entries = []
        for batch_num, batch in enumerate(batches, 1):
            # update progress bar by the portion this batch contributes
            pbar.update(batch_weight)
            pbar.set_description(f"#{index+1}: batch -> {batch_num}/{total_batches}")

            result = translate_batch(client, batch, filename, batch_num, total_batches)

            if result is None:
                print(f"\n  Failed: Could not translate batch {batch_num}")
                return False

            # Build translated entries: keep original timestamps, use translated text
            translated_map = {item["index"]: item["text"] for item in result}
            for entry in batch:
                translated_entries.append(
                    {
                        "index": entry["index"],
                        "start": entry["start"],
                        "end": entry["end"],
                        "text": translated_map.get(entry["index"], entry["text"]),
                    }
                )

            # Rate limit delay between batches (skip after the last one)
            if batch_num < total_batches:
                time.sleep(2)

        # Reassemble and save the translated SRT
        translated_srt = build_srt_from_entries(translated_entries)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(translated_srt)

        return True

    except Exception as e:
        print(f"  Failed: {e}")
        return False


def main():
    """Main function to batch process all SRT files."""

    # Initialize OpenAI client (pointed at Groq)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found in .env file")
        print("Please create a .env file with your API key")
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    # Create folders if they don't exist
    Path(INPUT_FOLDER).mkdir(exist_ok=True)
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

    # Find all SRT files
    srt_files = list(Path(INPUT_FOLDER).glob("*.srt"))

    if not srt_files:
        print(f"No .srt files found in '{INPUT_FOLDER}/' folder")
        print(f"Please add your English SRT files to the '{INPUT_FOLDER}/' folder")
        return

    print("SRT Batch Translator")
    print("=" * 50)
    print(f"Input folder:  {INPUT_FOLDER}/")
    print(f"Output folder: {OUTPUT_FOLDER}/")
    print(f"Files found:   {len(srt_files)}")
    print(f"Model:         {MODEL}")
    print(f"Batch size:    {BATCH_SIZE}")
    print("=" * 50)

    # Process files
    successful = 0
    failed = 0
    pbar = tqdm(total=100)
    each_file_weight = 100 / len(srt_files)
    # ensure we close the progress bar when done
    for index, srt_file in enumerate(srt_files):
        # show which file is being processed
        pbar.set_description(f"Translating file #{index + 1}")
        output_filename = srt_file.stem + "_DE.srt"
        output_path = Path(OUTPUT_FOLDER) / output_filename
        if process_srt_file(
            client, index, pbar, each_file_weight, srt_file, output_path
        ):
            successful += 1
        else:
            failed += 1

        time.sleep(1)

    pbar.close()

    # Summary
    print("\n" + "=" * 50)
    print("Translation Complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed:     {failed}")
    print(f"  Output:     {OUTPUT_FOLDER}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
