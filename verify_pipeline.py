import sys
import time
import requests

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = 'http://127.0.0.1:8000'

def test():
    print("Testing connection to dashboard...")
    r = requests.get(f"{BASE}/api/scenarios")
    print("Scenarios Available:", list(r.json()['scenarios'].keys()))

    for sc in ['normal_day', 'night_rain_rare', 'dirty_lens_artifact', 'frozen_camera']:
        requests.post(f"{BASE}/api/set-scenario", json={"scenario": sc})
        time.sleep(2.0)
        r = requests.get(f"{BASE}/api/telemetry")
        d = r.json()
        print("\n" + "=" * 55)
        print(f"Scenario: {sc}")
        print(f"Frame ID: {d.get('frame_id')} | FPS: {d.get('fps')}")
        print(f"Primary Decision: {d.get('primary_decision')} (Confidence: {d.get('primary_confidence')})")
        ch = d.get('camera_health', {})
        print(f"Camera Health: {ch.get('overall_health')} | Blur: {ch.get('blur_health')} | Occl: {ch.get('occlusion_health')}")
        print(f"Diagnostic Tags: {ch.get('status_tags')}")
        print(f"Active Tracks: {len(d.get('tracks', []))}")
        if d.get('predictions'):
            p = d['predictions'][0]
            print(f"Prediction: {p.get('prediction')} | Probabilities: {p.get('probabilities')}")
            print(f"AI Reasoning: {p.get('explanation')}")

if __name__ == "__main__":
    test()
