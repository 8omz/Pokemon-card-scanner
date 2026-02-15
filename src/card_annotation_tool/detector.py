import cv2
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class CardDetector:
    def __init__(self, debug=False):
        self.debug = debug
        
    def order_points(self, pts):
        """
        Orders points in [TL, TR, BR, BL] order.
        """
        rect = np.zeros((4, 2), dtype="float32")
        
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect

    def detect_card(self, image):
        """
        Detects the largest quadrilateral contour in the image.
        Returns:
            corners (np.array): 4 points [TL, TR, BR, BL] or None if not found.
            debug_img (np.array): Image with drawn contours for debugging.
        """
        if image is None:
            return None, None
            
        original = image.copy()
        
        # Resize for speed/consistency (work on a fixed scale, map back later)
        # But for high precision corners, working on full res is better if not too slow.
        # Let's work on full res for now, assuming images are reasonable (e.g. 1080p).
        # Actually, blurring is kernel dependent, so resizing helps consistency.
        # Let's try full res first.
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Blur to remove noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge/Threshold Detection
        # Use Canny with dilation to find the border edges
        # This is often better when contrast is low but edges exist
        
        # Auto-Canny parameters
        v = np.median(blurred)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        
        edged = cv2.Canny(blurred, lower, upper)
        
        # Dilate to connect broken edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.dilate(edged, kernel, iterations=1)
                                     
        # Find Contours
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort by area (largest first)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        
        if self.debug and len(cnts) > 0:
             print(f"  Debug: Top {len(cnts)} contours areas relative to image:")
             for i, c in enumerate(cnts):
                 print(f"    #{i}: {cv2.contourArea(c) / (original.shape[0]*original.shape[1]):.2f}")
                 
             # Save thresh for debug
             if not os.path.exists("data/reconstructed_crops/debug_mask.jpg"):
                 cv2.imwrite("data/reconstructed_crops/debug_mask.jpg", thresh)
        
        screen_cnt = None
        
        for c in cnts:
            # Approximate the contour
            peri = cv2.arcLength(c, True)
            # 2% of perimeter is a good approximation accuracy
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            # If our approximated contour has 4 points, we can assume it's the card
            if len(approx) == 4:
                screen_cnt = approx
                break
                
        if screen_cnt is None:
            # Fallback: finding the bounding rect of the larges contour might be better than nothing
            if len(cnts) > 0:
                 rect = cv2.minAreaRect(cnts[0])
                 box = cv2.boxPoints(rect)
                 screen_cnt = np.int32(box)
            else:
                 return None, None

        # Check filter conditions
        h, w = image.shape[:2]
        contour_area = cv2.contourArea(screen_cnt)
        image_area = w * h
        
        if contour_area < 0.10 * image_area:
             if self.debug: print(f"  Debug: Rejected area {contour_area/image_area:.2f} (Too small)")
             return None, None
             
        # Check if touching borders (within 5 pixels)
        # If all 4 points are on the border, it is likely the frame
        is_frame = True
        margin = 5
        pts_on_border = 0
        for pt in screen_cnt.reshape(4, 2):
            px, py = pt
            if px < margin or px > w - margin or py < margin or py > h - margin:
                pts_on_border += 1
        
        if pts_on_border >= 3:
             if self.debug: print(f"  Debug: Rejected (Touching border: {pts_on_border} pts)")
             return None, None
             
        if contour_area > 0.99 * image_area:
             if self.debug: print(f"  Debug: Rejected area {contour_area/image_area:.2f} (Too massive)")
             return None, None

        # Reshape to (4, 2)
        pts = screen_cnt.reshape(4, 2)
        ordered_pts = self.order_points(pts)
        
        # Create debug image
        debug_img = original.copy()
        cv2.drawContours(debug_img, [screen_cnt.astype(int)], -1, (0, 255, 0), 3)
        
        # Draw corners
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)] # BGR: Blue, Green, Red, Cyan
        labels = ["TL", "TR", "BR", "BL"]
        for i, (x, y) in enumerate(ordered_pts):
            cv2.circle(debug_img, (int(x), int(y)), 10, colors[i], -1)
            cv2.putText(debug_img, labels[i], (int(x)+10, int(y)+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return ordered_pts, debug_img

if __name__ == "__main__":
    # Test on dataset images
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_csv = os.path.join(project_root, "data", "manifests", "dataset.csv")
    reconstructed_dir = os.path.join(project_root, "data", "reconstructed_crops")
    if not os.path.exists(reconstructed_dir):
        os.makedirs(reconstructed_dir)
        
    import csv
    
    detector = CardDetector(debug=True)
    
    if os.path.exists(dataset_csv):
        print(f"Testing detector on images from {dataset_csv}...")
        with open(dataset_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            seen_images = set()
            count = 0
            
            for row in rows:
                img_path = row['image_path']
                if img_path in seen_images:
                    continue
                seen_images.add(img_path)
                
                if not os.path.exists(img_path):
                    continue
                    
                image = cv2.imread(img_path)
                print(f"Detecting: {os.path.basename(img_path)}...")
                corners, debug_img = detector.detect_card(image)
                
                if corners is not None:
                    # Save debug image
                    out_path = os.path.join(reconstructed_dir, f"debug_det_{os.path.basename(img_path)}")
                    cv2.imwrite(out_path, debug_img)
                    print(f"  Success! Saved debug to {out_path}")
                    
                    gt_corners = np.array([
                        [float(row['corner_tl_x']), float(row['corner_tl_y'])],
                        [float(row['corner_tr_x']), float(row['corner_tr_y'])],
                        [float(row['corner_br_x']), float(row['corner_br_y'])],
                        [float(row['corner_bl_x']), float(row['corner_bl_y'])]
                    ])
                    
                    dist = 0
                    for i in range(4):
                        d = np.linalg.norm(corners[i] - gt_corners[i])
                        dist += d
                    avg_dist = dist / 4
                    print(f"  Avg Deviation from GT: {avg_dist:.2f} px")
                else:
                    print("  Failed to detect (or filtered out).")
                
                count += 1
                if count >= 10:
                    break
