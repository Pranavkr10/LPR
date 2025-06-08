import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import os
import numpy as np

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Suite")

        #Variables
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.current_image = None
        self.cropping_vars = None
        self.noise_vars = None
        self.recent_files = []
        self.max_recent_files = 5

        #Notebook
        self.notebook = ttk.Notebook(root)
        self.tab_main = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_main, text="Main Controls")
        self.notebook.pack(expand=1, fill="both")

        #Directory Selection
        ttk.Label(self.tab_main, text="Input Directory:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(self.tab_main, textvariable=self.input_dir, width=40).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.tab_main, text="Browse", command=self.browse_input_dir).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(self.tab_main, text="Output Directory:").grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(self.tab_main, textvariable=self.output_dir, width=40).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self.tab_main, text="Browse", command=self.browse_output_dir).grid(row=1, column=2, padx=5, pady=5)

        #Actions
        ttk.Button(self.tab_main, text="Crop Images", command=self.open_cropping_interface).grid(row=2, column=0, pady=10)
        ttk.Button(self.tab_main, text="Preprocess Images", command=self.run_preprocessing).grid(row=2, column=1, pady=10)
        ttk.Button(self.tab_main, text="Invert Colours", command=self.run_inversion).grid(row=2, column=2, pady=10)
        ttk.Button(self.tab_main, text="Manual Noise Cleaning", command=self.open_noise_cleaner).grid(row=3, column=0, pady=10)

        #Recent Files
        ttk.Label(self.tab_main, text="Recent Files:").grid(row=4, column=0, columnspan=3, pady=(10, 0), sticky='w')
        self.recent_list = tk.Listbox(self.tab_main, height=5)
        self.recent_list.grid(row=5, column=0, columnspan=3, padx=5, pady=(0, 10), sticky='nsew')

        self.tab_main.grid_rowconfigure(5, weight=1)
        self.tab_main.grid_columnconfigure(1, weight=1)

    def add_to_recent(self, file_path):
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        self.recent_files.insert(0, file_path)
        if len(self.recent_files) > self.max_recent_files:
            self.recent_files = self.recent_files[:self.max_recent_files]
        self.recent_list.delete(0, tk.END)
        for f in self.recent_files:
            self.recent_list.insert(tk.END, os.path.basename(f))

    def browse_input_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.input_dir.set(path)

    def browse_output_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    def valid_image(self, filename):
        return filename.lower().endswith(('.png', '.jpg', '.jpeg'))

    #Cropping 
    def open_cropping_interface(self):
        crop_window = tk.Toplevel(self.root)
        crop_window.title("Image Cropping")
        self.cropping_vars = {'cropping': False, 'regions': [], 'current_image': None, 'clone': None, 'image_path': None}

        ttk.Button(crop_window, text="Select Image", command=self.load_image_for_cropping).pack(pady=5)
        ttk.Button(crop_window, text="Save Crops", command=self.save_cropped_regions).pack(pady=5)

    def load_image_for_cropping(self):
        file_path = filedialog.askopenfilename(initialdir=self.input_dir.get())
        if not file_path:
            return
        self.add_to_recent(file_path)
        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Error", "Cannot open image.")
            return
        self.cropping_vars.update({
            'current_image': image.copy(),
            'clone': image.copy(),
            'regions': [],
            'image_path': file_path
        })

        cv2.namedWindow("Image Cropper", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Image Cropper", self.mouse_crop)

        while True:
            cv2.imshow("Image Cropper", self.cropping_vars['current_image'])
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                break
            elif key == ord('u'):
                self.cropping_vars['current_image'] = self.cropping_vars['clone'].copy()
                self.cropping_vars['regions'].clear()
        cv2.destroyAllWindows()

    def mouse_crop(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.cropping_vars['x_start'], self.cropping_vars['y_start'] = x, y
            self.cropping_vars['cropping'] = True
        elif event == cv2.EVENT_LBUTTONUP:
            x1, y1 = self.cropping_vars['x_start'], self.cropping_vars['y_start']
            x2, y2 = x, y
            self.cropping_vars['cropping'] = False
            self.cropping_vars['regions'].append((min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)))
            cv2.rectangle(self.cropping_vars['current_image'], (x1, y1), (x2, y2), (0, 255, 0), 2)

    def save_cropped_regions(self):
        output_dir = self.output_dir.get()
        os.makedirs(output_dir, exist_ok=True)
        img_path = self.cropping_vars.get('image_path', '')
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        for i, (x1, y1, x2, y2) in enumerate(self.cropping_vars['regions']):
            cropped = self.cropping_vars['clone'][y1:y2, x1:x2]
            if cropped.size > 0:
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_region_{i}.png"), cropped)
        messagebox.showinfo("Info", f"Saved {len(self.cropping_vars['regions'])} crops.")

    #Preprocessing
    def run_preprocessing(self):
        input_dir, output_dir = self.input_dir.get(), self.output_dir.get()
        os.makedirs(output_dir, exist_ok=True)

        for img_name in os.listdir(input_dir):
            if not self.valid_image(img_name):
                continue
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            if np.mean(img) > 127:
                img = cv2.bitwise_not(img)

            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            cv2.imwrite(os.path.join(output_dir, img_name), img)
        messagebox.showinfo("Info", "Preprocessing completed!")

    #Inversion 
    def run_inversion(self):
        input_dir, output_dir = self.input_dir.get(), self.output_dir.get()
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            if not self.valid_image(filename):
                continue
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            inverted = cv2.bitwise_not(img)
            cv2.imwrite(os.path.join(output_dir, filename), inverted)
        messagebox.showinfo("Info", "Color inversion completed!")

    #ManualNoiseCleaning 
    def open_noise_cleaner(self):
        file_path = filedialog.askopenfilename(initialdir=self.input_dir.get())
        if not file_path:
            return

        self.add_to_recent(file_path)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("Error", "Image can't be loaded.")
            return

        h, w = img.shape
        self.noise_vars = {
            'undo_stack': [],
            'scale_factor': 3,
            'current_image': cv2.resize(img, (w*3, h*3), interpolation=cv2.INTER_NEAREST)
        }
        self.noise_vars['undo_stack'].append(self.noise_vars['current_image'].copy())

        cv2.namedWindow("Noise Cleaning", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Noise Cleaning", self.noise_clean_callback)

        while True:
            cv2.imshow("Noise Cleaning", self.noise_vars['current_image'])
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            elif key == ord('s'):
                self.save_cleaned_image(file_path)
                break
            elif key == ord('u') and len(self.noise_vars['undo_stack']) > 1:
                self.noise_vars['undo_stack'].pop()
                self.noise_vars['current_image'] = self.noise_vars['undo_stack'][-1].copy()
        cv2.destroyAllWindows()

    def noise_clean_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.noise_vars['undo_stack'].append(self.noise_vars['current_image'].copy())
            cv2.circle(self.noise_vars['current_image'], (x, y), 10, 0, -1)
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(self.noise_vars['current_image'], (x, y), 10, 0, -1)

    def save_cleaned_image(self, original_path):
        output_dir = self.output_dir.get()
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(original_path)
        name, ext = os.path.splitext(filename)
        cleaned = cv2.resize(self.noise_vars['current_image'],
                             (self.noise_vars['current_image'].shape[1] // 3,
                              self.noise_vars['current_image'].shape[0] // 3),
                             interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(output_dir, f"{name}_cleaned{ext}"), cleaned)
        messagebox.showinfo("Info", "Cleaned image saved!")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()