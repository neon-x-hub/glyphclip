import urllib.request
import json
from pathlib import Path

def test_query(image_path, query_text=None):
    boundary = "----BoundaryTypographic"
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{Path(image_path).name}\"\r\nContent-Type: image/png\r\n\r\n".encode("utf-8"),
        img_bytes,
        f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"top_k\"\r\n\r\n5\r\n".encode("utf-8"),
    ]
    if query_text:
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"query_text\"\r\n\r\n{query_text}\r\n".encode("utf-8"))
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))

    body = b"".join(parts)
    req = urllib.request.Request(
        "http://127.0.0.1:8080/v1/recognize",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print(f"\nQuery: {image_path} (Text hint: {query_text})")
        print(f"Latency: {res['latency_ms']} ms | Detected Glyphs: {res['characters_detected']}")
        for p in res["predictions"]:
            print(f"  #{p['rank']}: {p['family']} ({p['confidence']*100:.1f}%)")

if __name__ == "__main__":
    test_query("funnel.png", query_text="Funnel")
    test_query("funnel.png", query_text=None)
