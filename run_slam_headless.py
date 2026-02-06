"""
SIFT Visual SLAM - Headless Version
บันทึกผลลัพธ์เป็นไฟล์แทนการแสดงหน้าต่าง
"""
import cv2
import numpy as np
import time
import os

class SimpleSLAM:
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=2000)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        self.K = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.trajectory = []
        self.current_pos = np.zeros((3, 1))
        
    def extract_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        kp, des = self.sift.detectAndCompute(gray, None)
        return kp, des
    
    def match_features(self, des1, des2):
        if des1 is None or des2 is None:
            return []
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good.append(m)
        return good
    
    def process(self, frame, prev_frame, prev_kp, prev_des):
        kp, des = self.extract_features(frame)
        
        if des is None or prev_des is None:
            return kp, des, 0
        
        matches = self.match_features(prev_des, des)
        
        if len(matches) > 8:
            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp[m.trainIdx].pt for m in matches])
            
            E, mask = cv2.findEssentialMat(pts1, pts2, self.K, cv2.RANSAC, 0.999, 1.0)
            
            if E is not None:
                _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
                self.current_pos += t * 0.1
                self.trajectory.append(self.current_pos.copy())
                return kp, des, len(matches)
        
        return kp, des, 0
    
    def draw_trajectory(self):
        canvas = np.zeros((600, 800, 3), dtype=np.uint8)
        
        if len(self.trajectory) > 1:
            for i in range(len(self.trajectory) - 1):
                x1 = int(np.clip(self.trajectory[i][0] * 100 + 400, 0, 799))
                y1 = int(np.clip(self.trajectory[i][2] * 100 + 300, 0, 599))
                x2 = int(np.clip(self.trajectory[i+1][0] * 100 + 400, 0, 799))
                y2 = int(np.clip(self.trajectory[i+1][2] * 100 + 300, 0, 599))
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            curr_x = int(np.clip(self.current_pos[0] * 100 + 400, 0, 799))
            curr_y = int(np.clip(self.current_pos[2] * 100 + 300, 0, 599))
            cv2.circle(canvas, (curr_x, curr_y), 8, (0, 0, 255), -1)
        
        return canvas

def main():
    print("=" * 50)
    print("   SIFT Visual SLAM - Headless Mode")
    print("=" * 50)
    print()
    print("โหมดนี้จะบันทึกผลลัพธ์เป็นไฟล์:")
    print("  - output_video.avi = วิดีโอจากกล้อง")  
    print("  - trajectory.png = เส้นทางการเคลื่อนที่")
    print()
    print("กำลังบันทึก... รอ 10 วินาที")
    print("-" * 50)
    
    slam = SimpleSLAM()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: ไม่สามารถเปิดกล้องได้!")
        return
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (640, 480))
    
    ret, prev_frame = cap.read()
    prev_frame = cv2.resize(prev_frame, (640, 480))
    prev_kp, prev_des = slam.extract_features(prev_frame)
    
    frame_count = 0
    start_time = time.time()
    duration = 10  # รัน 10 วินาที
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 480))
        kp, des, num_matches = slam.process(frame, prev_frame, prev_kp, prev_des)
        
        # Draw info on frame
        cv2.putText(frame, f"Features: {len(kp)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Matches: {num_matches}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Trajectory: {len(slam.trajectory)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write to video
        out.write(frame)
        
        prev_frame = frame.copy()
        prev_kp, prev_des = kp, des
        frame_count += 1
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f"Frame {frame_count} | Matches: {num_matches} | Trajectory: {len(slam.trajectory)} pts | Time: {elapsed:.1f}s")
        
        # Stop after duration
        if time.time() - start_time > duration:
            break
    
    # Cleanup
    cap.release()
    out.release()
    
    # Save trajectory image
    trajectory_img = slam.draw_trajectory()
    cv2.putText(trajectory_img, "Camera Trajectory (Top View)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(trajectory_img, f"Total points: {len(slam.trajectory)}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.imwrite('trajectory.png', trajectory_img)
    
    print()
    print("=" * 50)
    print("   เสร็จสิ้น!")
    print("=" * 50)
    print(f"Processed {frame_count} frames")
    print(f"Trajectory points: {len(slam.trajectory)}")
    print()
    print("ไฟล์ที่สร้าง:")
    print(f"  1. output_video.avi ({os.path.getsize('output_video.avi') / 1024:.1f} KB)")
    print(f"  2. trajectory.png")
    print()
    print("เปิดไฟล์ดูผลลัพธ์ได้เลยครับ!")

if __name__ == "__main__":
    main()
