"""
SIFT Visual SLAM - Live Camera View
แก้ปัญหาหน้าต่างไม่แสดง
"""
import cv2
import numpy as np

print("=" * 50)
print("   SIFT Visual SLAM - Live Camera")
print("=" * 50)
print()

# สร้างหน้าต่างก่อน
cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
cv2.namedWindow('Trajectory', cv2.WINDOW_NORMAL)

# ย้ายหน้าต่างไปตำแหน่งที่เห็นชัด
cv2.moveWindow('Camera', 100, 100)
cv2.moveWindow('Trajectory', 750, 100)

# Resize หน้าต่าง
cv2.resizeWindow('Camera', 640, 480)
cv2.resizeWindow('Trajectory', 640, 480)

print("สร้างหน้าต่างเรียบร้อย!")
print("ถ้าไม่เห็น ลองกด Alt+Tab หรือดู Taskbar")
print()

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
                self.trajectory.append(self.current_pos.flatten().copy())
                return kp, des, len(matches)
        
        return kp, des, 0
    
    def draw_trajectory(self):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw grid
        for i in range(0, 640, 80):
            cv2.line(canvas, (i, 0), (i, 480), (30, 30, 30), 1)
        for i in range(0, 480, 80):
            cv2.line(canvas, (0, i), (640, i), (30, 30, 30), 1)
        
        if len(self.trajectory) > 1:
            for i in range(len(self.trajectory) - 1):
                x1 = int(np.clip(self.trajectory[i][0] * 100 + 320, 0, 639))
                y1 = int(np.clip(self.trajectory[i][2] * 100 + 240, 0, 479))
                x2 = int(np.clip(self.trajectory[i+1][0] * 100 + 320, 0, 639))
                y2 = int(np.clip(self.trajectory[i+1][2] * 100 + 240, 0, 479))
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Current position
            curr_x = int(np.clip(self.trajectory[-1][0] * 100 + 320, 0, 639))
            curr_y = int(np.clip(self.trajectory[-1][2] * 100 + 240, 0, 479))
            cv2.circle(canvas, (curr_x, curr_y), 8, (0, 0, 255), -1)
        
        # Center cross
        cv2.drawMarker(canvas, (320, 240), (100, 100, 100), cv2.MARKER_CROSS, 20, 1)
        
        return canvas

def main():
    slam = SimpleSLAM()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: ไม่สามารถเปิดกล้องได้!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    ret, prev_frame = cap.read()
    if not ret:
        print("ERROR: ไม่สามารถอ่าน frame!")
        return
        
    prev_kp, prev_des = slam.extract_features(prev_frame)
    
    print("กล้องพร้อมใช้งาน!")
    print("กด Q เพื่อหยุด")
    print("-" * 50)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        kp, des, num_matches = slam.process(frame, prev_frame, prev_kp, prev_des)
        
        # Draw keypoints
        frame_display = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0), 
                                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Draw info
        cv2.rectangle(frame_display, (5, 5), (250, 100), (0, 0, 0), -1)
        cv2.putText(frame_display, f"Features: {len(kp)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_display, f"Matches: {num_matches}", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame_display, f"Trajectory: {len(slam.trajectory)}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw trajectory
        trajectory_canvas = slam.draw_trajectory()
        cv2.putText(trajectory_canvas, "Camera Trajectory (Top View)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(trajectory_canvas, "Press Q to quit", (10, 460),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Show windows
        cv2.imshow('Camera', frame_display)
        cv2.imshow('Trajectory', trajectory_canvas)
        
        # Update
        prev_frame = frame.copy()
        prev_kp, prev_des = kp, des
        frame_count += 1
        
        # Print progress
        if frame_count % 60 == 0:
            print(f"Frame {frame_count} | Matches: {num_matches} | Points: {len(slam.trajectory)}")
        
        # Check key - IMPORTANT: waitKey must be called!
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("User quit")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal: {frame_count} frames, {len(slam.trajectory)} trajectory points")

if __name__ == "__main__":
    main()
