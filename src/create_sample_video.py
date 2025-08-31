# src/create_sample_video.py
import os, cv2, numpy as np
os.makedirs("data", exist_ok=True)
out_path = "data/sample_video.mp4"
w, h = 320, 240
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, 15, (w,h))
for i in range(150):
    canvas = np.zeros((h,w,3), dtype=np.uint8) + 255
    # draw moving square that sometimes is anomalous-looking
    x = int((w-40) * (0.5 + 0.5 * np.sin(i*0.06)))
    y = int((h-40) * (0.5 + 0.5 * np.cos(i*0.05)))
    cv2.rectangle(canvas, (x,y), (x+30, y+30), (0,0,255), -1)
    # add random noise occasionally
    if i % 37 == 0:
        noise = (np.random.randn(h,w,3) * 40).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    writer.write(canvas)
writer.release()
print("Sample video created at:", out_path)
