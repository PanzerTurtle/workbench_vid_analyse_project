if __name__ == "__main__":
    import os
    import json
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from _analysis import analyze_frame_worker
    from _file_explorer import select_file
    from _frame_extract import *
    from _utility_function import read_num_in_filename, log_parse_error
    from _write_reports import json2html_convert_all
    from _config import EXTRACTED_FRAMES_DIR, FRAME_INTERVAL, PARSE_ERROR_LOG, OUTPUT_JSON, OUTPUT_HTML
    
    INPUT_VIDEO = select_file(INITIAL_DIR="./vid_samples")
    # INPUT_VIDEO = "./vid_samples/night_front_door.mp4"
    
    start = time.perf_counter()

    print("=" * 60)
    print("VIDEO ANALYSIS PIPELINE")
    print("=" * 60)

    print("\n[1] Extracting frames...")
    timestamps = extract_frames(INPUT_VIDEO, EXTRACTED_FRAMES_DIR, interval_sec=FRAME_INTERVAL)

    print("\n[2] Analyzing frames...")
    frames = sorted(
        (f for f in os.listdir(EXTRACTED_FRAMES_DIR) if f.endswith(".jpg")),
        key=read_num_in_filename,
    )

    events = []
    errors = 0

    # multiprocessing analysis
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(analyze_frame_worker, f, timestamps, EXTRACTED_FRAMES_DIR): f for f in frames
        }

        for i, future in enumerate(as_completed(futures), 1):
            frame = futures[future]
            parsed, error = future.result()

            if parsed:
                events.append(parsed)
                print(f"[OK] {i}/{len(frames)} {frame}")
            else:
                errors += 1
                log_parse_error(frame, error, output=PARSE_ERROR_LOG)
                print(f"[FAIL] {i}/{len(frames)} {frame}")

    events.sort(key=lambda e: e["timestamp"])

    with open(OUTPUT_JSON, "w") as f:
        json.dump(events, f, indent=2)

    print("\n[3] Generating report...")
    json2html_convert_all(OUTPUT_JSON, OUTPUT_HTML)

    elapsed = time.perf_counter() - start
    print("\nCompleted")
    print(f"Frames analyzed: {len(events)}")
    print(f"Errors: {errors}")
    print(f"Total time: {elapsed:.2f}s ({elapsed/60:.1f} min)")
