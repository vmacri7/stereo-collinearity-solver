import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import csv
import os

class PointLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Correspondence Labeler")
        
        # load images
        self.img1 = cv2.imread("425.tif")
        self.img2 = cv2.imread("426.tif")
        
        if self.img1 is None or self.img2 is None:
            print("error: could not load images")
            return
            
        # convert to rgb for display
        self.img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB)
        self.img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)
        
        # store original images
        self.orig_img1 = self.img1.copy()
        self.orig_img2 = self.img2.copy()
        
        # initialize variables
        self.points1 = []
        self.points2 = []
        self.current_point = 1
        self.zoom_factor = 1.0
        self.pan_x1 = 0
        self.pan_y1 = 0
        self.pan_x2 = 0
        self.pan_y2 = 0
        self.waiting_for_second_point = False

        # status bar
        self.status_bar = ttk.Label(root, text="point 1/100")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # load existing points if available
        self.load_existing_points()
        
        # create main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # create image frames
        self.frame1 = ttk.Frame(self.main_frame)
        self.frame1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.frame2 = ttk.Frame(self.main_frame)
        self.frame2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # create canvases
        self.canvas1 = tk.Canvas(self.frame1)
        self.canvas1.pack(fill=tk.BOTH, expand=True)
        
        self.canvas2 = tk.Canvas(self.frame2)
        self.canvas2.pack(fill=tk.BOTH, expand=True)
        
        # bind events
        self.canvas1.bind("<ButtonPress-1>", self.on_click1)
        self.canvas2.bind("<ButtonPress-1>", self.on_click2)
        
        # bind mouse wheel for zoom
        self.canvas1.bind("<MouseWheel>", self.zoom)
        self.canvas2.bind("<MouseWheel>", self.zoom)
        
        # bind middle mouse button for panning
        self.canvas1.bind("<Button-2>", self.start_pan1)
        self.canvas1.bind("<B2-Motion>", self.pan1)
        self.canvas2.bind("<Button-2>", self.start_pan2)
        self.canvas2.bind("<B2-Motion>", self.pan2)
        
        # bind 'c' key for point removal
        self.root.bind('c', self.remove_last_point)
        
        # initial display
        self.update_display()
        
    def load_existing_points(self):
        if os.path.exists('points.csv'):
            with open('points.csv', 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.points1.append((int(row['425_x']), int(row['425_y'])))
                    self.points2.append((int(row['426_x']), int(row['426_y'])))
                    self.current_point = len(self.points1) + 1
                    
            if self.current_point > 1:
                print(f"loaded {self.current_point - 1} existing points")
                self.status_bar.config(text=f"point {self.current_point-1}/100")
        
    def remove_last_point(self, event=None):
        if self.waiting_for_second_point:
            if len(self.points1) > 0:
                self.points1.pop()
                self.waiting_for_second_point = False
                self.status_bar.config(text=f"point {self.current_point}/100")
        else:
            if len(self.points2) > 0:
                self.points2.pop()
                self.waiting_for_second_point = True
                self.status_bar.config(text=f"click corresponding point in right image - point {self.current_point}/100")
        self.update_display()
        self.save_points()
            
    def start_pan1(self, event):
        self.pan_start_x1 = event.x
        self.pan_start_y1 = event.y
        
    def start_pan2(self, event):
        self.pan_start_x2 = event.x
        self.pan_start_y2 = event.y
            
    def pan1(self, event):
        dx = event.x - self.pan_start_x1
        dy = event.y - self.pan_start_y1
        self.pan_x1 += dx
        self.pan_y1 += dy
        self.pan_start_x1 = event.x
        self.pan_start_y1 = event.y
        self.update_display()
        
    def pan2(self, event):
        dx = event.x - self.pan_start_x2
        dy = event.y - self.pan_start_y2
        self.pan_x2 += dx
        self.pan_y2 += dy
        self.pan_start_x2 = event.x
        self.pan_start_y2 = event.y
        self.update_display()
        
    def zoom(self, event):
        if event.delta > 0:
            self.zoom_factor *= 1.1
        else:
            self.zoom_factor /= 1.1
        self.update_display()
        
    def update_display(self):
        # resize images according to zoom
        h1, w1 = self.orig_img1.shape[:2]
        h2, w2 = self.orig_img2.shape[:2]
        
        new_w1 = int(w1 * self.zoom_factor)
        new_h1 = int(h1 * self.zoom_factor)
        new_w2 = int(w2 * self.zoom_factor)
        new_h2 = int(h2 * self.zoom_factor)
        
        img1_resized = cv2.resize(self.orig_img1, (new_w1, new_h1))
        img2_resized = cv2.resize(self.orig_img2, (new_w2, new_h2))
        
        # draw points
        for i, (x, y) in enumerate(self.points1):
            scaled_x = int(x * self.zoom_factor)
            scaled_y = int(y * self.zoom_factor)
            cv2.circle(img1_resized, (scaled_x, scaled_y), 3, (0, 255, 0), -1)
            cv2.putText(img1_resized, str(i+1), (scaled_x+5, scaled_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        for i, (x, y) in enumerate(self.points2):
            scaled_x = int(x * self.zoom_factor)
            scaled_y = int(y * self.zoom_factor)
            cv2.circle(img2_resized, (scaled_x, scaled_y), 3, (0, 255, 0), -1)
            cv2.putText(img2_resized, str(i+1), (scaled_x+5, scaled_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # convert to PhotoImage
        self.photo1 = ImageTk.PhotoImage(Image.fromarray(img1_resized))
        self.photo2 = ImageTk.PhotoImage(Image.fromarray(img2_resized))
        
        # update canvases
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        
        self.canvas1.create_image(self.pan_x1, self.pan_y1, image=self.photo1, anchor="nw")
        self.canvas2.create_image(self.pan_x2, self.pan_y2, image=self.photo2, anchor="nw")
        
    def on_click1(self, event):
        if self.current_point > 100 or self.waiting_for_second_point:
            return
            
        x = int((event.x - self.pan_x1) / self.zoom_factor)
        y = int((event.y - self.pan_y1) / self.zoom_factor)
        self.points1.append((x, y))
        self.waiting_for_second_point = True
        self.status_bar.config(text=f"click corresponding point in right image - point {self.current_point}/100")
        self.update_display()
        
    def on_click2(self, event):
        if self.current_point > 100 or not self.waiting_for_second_point:
            return
            
        x = int((event.x - self.pan_x2) / self.zoom_factor)
        y = int((event.y - self.pan_y2) / self.zoom_factor)
        self.points2.append((x, y))
        self.waiting_for_second_point = False
        self.current_point += 1
        self.status_bar.config(text=f"point {self.current_point}/100")
        self.save_points()
        self.update_display()
        
    def save_points(self):
        with open('points.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['point_id', '425_x', '425_y', '426_x', '426_y'])
            for i in range(len(self.points2)):  # use points2 as it will always be <= points1
                writer.writerow([i+1, 
                               self.points1[i][0], 
                               self.points1[i][1],
                               self.points2[i][0], 
                               self.points2[i][1]])

if __name__ == "__main__":
    root = tk.Tk()
    app = PointLabeler(root)
    root.mainloop()