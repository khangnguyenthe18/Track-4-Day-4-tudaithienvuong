import os
import sys
import uvicorn

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def main():
    print("=" * 70)
    print("[Antigravity AI] Rare Pedestrian Detection with Camera Health Awareness")
    print("=" * 70)
    print("AI Stack: YOLOv8 + ByteTrack + Camera Health Engine + Rare Case Classifier")
    print("Serving interactive web dashboard on http://localhost:8000")
    print("Press Ctrl+C to terminate.")
    print("=" * 70)

    # Check if ML weights exist, if not, train initial model
    weights_path = os.path.join(BASE_DIR, "weights", "rare_case_model.pkl")
    if not os.path.exists(weights_path):
        print("\n[Init] Pretraining Random Forest classifier on synthetic feature distributions...")
        from src.rare_case.train import train_and_save_model
        train_and_save_model(weights_path)
        print("[Init] Model ready!\n")

    uvicorn.run("src.dashboard.app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")

if __name__ == "__main__":
    main()
