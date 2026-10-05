import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from src.data.synthetic_generator import SyntheticVideoGenerator, SCENARIOS

def main():
    print("=" * 65)
    print("[Generator] Generating Synthetic Simulated Traffic Benchmark Videos")
    print("=" * 65)
    output_dir = os.path.join(BASE_DIR, "data", "synthetic_videos")
    os.makedirs(output_dir, exist_ok=True)

    gen = SyntheticVideoGenerator(width=854, height=480, fps=20)
    for name, meta in SCENARIOS.items():
        out_file = os.path.join(output_dir, f"{name}.mp4")
        print(f"\n>> Generating {meta['title']}...")
        gen.generate_video(name, out_file, num_frames=120)
        print(f"   Saved: {out_file}")

    print("\n[Success] All 4 benchmark video scenarios generated successfully!")

if __name__ == "__main__":
    main()
