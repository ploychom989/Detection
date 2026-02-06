import cv2
import numpy as np

print("=== Camera Test ===")
print("กำลังเปิดกล้อง...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: ไม่สามารถเปิดกล้องได้!")
    exit()

print("กล้องเปิดสำเร็จ!")
print("กด ESC หรือ Q เพื่อหยุด")
print("-" * 30)

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error reading frame")
        break
    
    frame_count += 1
    
    # แสดง info ทุก 30 frames
    if frame_count % 30 == 0:
        print(f"Frame {frame_count} - Shape: {frame.shape}")
    
    # ใส่ข้อความบน frame
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press Q or ESC to quit", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # แสดงหน้าต่าง
    cv2.imshow('Camera Test - Press Q to quit', frame)
    
    # รอ key
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # Q หรือ ESC
        print("User pressed quit")
        break

print(f"\nTotal frames: {frame_count}")
cap.release()
cv2.destroyAllWindows()
print("Done!")
