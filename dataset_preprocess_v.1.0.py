import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import os
import numpy as np
from PIL import Image, ImageTk

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
        self.processing_window = None
        self.recent_files = []
        self.max_recent_files = 5
        
        #SetupUI
        self.create_main_interface()
        self.setup_recent_files_list()

    def create_main_interface(self):
        self.notebook = ttk.Notebook(self.root)
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

        
        ttk.Button(self.tab_main, text="Process Images", command=self.open_cropping_interface).grid(row=2, column=0, columnspan=3, pady=10)

    def setup_recent_files_list(self):
        ttk.Label(self.tab_main, text="Recent Files:").grid(row=3, column=0, columnspan=3, pady=(10, 0), sticky='w')
        self.recent_list = tk.Listbox(self.tab_main, height=5)
        self.recent_list.grid(row=4, column=0, columnspan=3, padx=5, pady=(0, 10), sticky='nsew')

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

    def open_cropping_interface(self):
        if not self.input_dir.get() or not os.path.isdir(self.input_dir.get()):
            messagebox.showerror("Error", "Please select a valid input directory first!")
            return

        self.crop_window = tk.Toplevel(self.root)
        self.crop_window.title("Image Cropping")
        self.cropping_vars = {
            'cropping': False,
            'regions': [],
            'current_image': None,
            'clone': None,
            'image_path': None
        }

        ttk.Button(self.crop_window, text="Select Image", command=self.load_image_for_cropping).pack(pady=5)
        ttk.Button(self.crop_window, text="Process Crops", command=self.prepare_processing).pack(pady=5)

    def load_image_for_cropping(self):
        file_path = filedialog.askopenfilename(initialdir=self.input_dir.get())
        if not file_path:
            return
        
        try:
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Failed to read image")
                
            self.add_to_recent(file_path)
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
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            cv2.destroyAllWindows()

    def mouse_crop(self, event, x, y, flags, param):
        try:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.cropping_vars['x_start'], self.cropping_vars['y_start'] = x, y
                self.cropping_vars['cropping'] = True
            elif event == cv2.EVENT_LBUTTONUP:
                x1, y1 = self.cropping_vars['x_start'], self.cropping_vars['y_start']
                x2, y2 = x, y
                self.cropping_vars['cropping'] = False
                self.cropping_vars['regions'].append((
                    min(x1, x2), min(y1, y2),
                    max(x1, x2), max(y1, y2)
                ))
                cv2.rectangle(self.cropping_vars['current_image'], (x1, y1), (x2, y2), (0, 255, 0), 2)
        except Exception as e:
            print(f"Cropping error: {str(e)}")

    def prepare_processing(self):
        try:
            if not self.cropping_vars or not self.cropping_vars['regions']:
                messagebox.showwarning("Warning", "No regions selected for processing!")
                return

            output_dir = self.output_dir.get()
            if not output_dir:
                messagebox.showwarning("Warning", "Please select an output directory first!")
                return

            os.makedirs(output_dir, exist_ok=True)
            cropped_images = []
            
            for (x1, y1, x2, y2) in self.cropping_vars['regions']:
                cropped = self.cropping_vars['clone'][y1:y2, x1:x2]
                if cropped.size > 0:
                    cropped_images.append(cropped.copy())

            if cropped_images:
                self.show_processing_interface(
                    cropped_images,
                    self.cropping_vars['image_path'],
                    output_dir
                )
            else:
                messagebox.showwarning("Warning", "No valid regions to process!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Processing preparation failed: {str(e)}")

    def show_processing_interface(self, cropped_images, img_path, output_dir):
        try:
            if self.processing_window:
                self.processing_window.destroy()

            self.processing_window = tk.Toplevel(self.root)
            self.processing_window.title("Process Cropped Images")
            
            self.processing_data = {
                'images': cropped_images,
                'output_dir': output_dir,
                'base_name': os.path.splitext(os.path.basename(img_path))[0],
                'inversion_flags': [tk.BooleanVar() for _ in cropped_images],
                'cleaned_images': [None] * len(cropped_images)
            }

            # Create thumbnail grid
            for idx, img in enumerate(cropped_images):
                frame = ttk.Frame(self.processing_window)
                frame.grid(row=idx // 3, column=idx % 3, padx=5, pady=5)

                # Convert to display format
                if len(img.shape) == 3:
                    display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    
                pil_img = Image.fromarray(display_img)
                pil_img.thumbnail((150, 150))
                photo = ImageTk.PhotoImage(pil_img)

                label = ttk.Label(frame, image=photo)
                label.image = photo
                label.pack()

                ttk.Checkbutton(frame, text="Invert", 
                              variable=self.processing_data['inversion_flags'][idx]).pack()
                ttk.Button(frame, text="Clean Noise", 
                          command=lambda i=idx: self.clean_image(i)).pack()

            ttk.Button(self.processing_window, text="Finalize Processing", 
                      command=self.finalize_processing).grid(row=999, column=0, columnspan=3, pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Interface creation failed: {str(e)}")

    def clean_image(self, index):
        try:
            img = self.processing_data['images'][index]
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            #Apply Otsu thresholding before cleaning
            _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            cleaned = self.manual_noise_clean(binary_img)
            if cleaned is not None:
                self.processing_data['cleaned_images'][index] = cleaned
                
        except Exception as e:
            messagebox.showerror("Error", f"Cleaning failed: {str(e)}")

    def manual_noise_clean(self, img):
        try:
            h, w = img.shape
            scale_factor = 3
            working_img = cv2.resize(img, (w*scale_factor, h*scale_factor), 
                                   interpolation=cv2.INTER_NEAREST)
            
            self.noise_vars = {
                'undo_stack': [working_img.copy()],
                'current_image': working_img,
                'scale_factor': scale_factor,
                'current_mode': 0  #0->black, 255-> white
            }

            cv2.namedWindow("Noise Cleaning", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("Noise Cleaning", self.noise_clean_callback)
            self.update_window_title()

            while True:
                cv2.imshow("Noise Cleaning", self.noise_vars['current_image'])
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  #ESC
                    cv2.destroyAllWindows()
                    return None
                elif key == ord('s'):
                    cleaned = cv2.resize(self.noise_vars['current_image'], 
                                       (w, h), 
                                       interpolation=cv2.INTER_NEAREST)
                    cv2.destroyAllWindows()
                    return cleaned
                elif key == ord('u') and len(self.noise_vars['undo_stack']) > 1:
                    self.noise_vars['undo_stack'].pop()
                    self.noise_vars['current_image'] = self.noise_vars['undo_stack'][-1].copy()
                elif key == ord('b'):  #Switch to black mode
                    self.noise_vars['current_mode'] = 0
                    self.update_window_title()
                elif key == ord('w'):  #Switch to white mode
                    self.noise_vars['current_mode'] = 255
                    self.update_window_title()
                    
        except Exception as e:
            messagebox.showerror("Error", f"Noise cleaning failed: {str(e)}")
            return None

    def update_window_title(self):
        mode = "Black" if self.noise_vars['current_mode'] == 0 else "White"
        cv2.setWindowTitle("Noise Cleaning", f"Noise Cleaning - Mode: {mode} (B/W to toggle)")

    def noise_clean_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.noise_vars['undo_stack'].append(self.noise_vars['current_image'].copy())
            color = self.noise_vars['current_mode']
            cv2.circle(self.noise_vars['current_image'], (x, y), 5, color, -1)
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            color = self.noise_vars['current_mode']
            cv2.circle(self.noise_vars['current_image'], (x, y), 5, color, -1)

    def finalize_processing(self):
        try:
            output_dir = self.processing_data['output_dir']
            os.makedirs(output_dir, exist_ok=True)
            
            for idx in range(len(self.processing_data['images'])):
                #Get the image to process
                if self.processing_data['cleaned_images'][idx] is not None:
                    img = self.processing_data['cleaned_images'][idx]
                else:
                    img = self.processing_data['images'][idx].copy()
                
                #Convert to grayscale if needed
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                #Apply inversion if selected
                if self.processing_data['inversion_flags'][idx].get():
                    img = cv2.bitwise_not(img)
                
                #Final processing
                img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
                _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                #Save result
                output_path = os.path.join(
                    output_dir,
                    f"{self.processing_data['base_name']}_region_{idx}.png"
                )
                cv2.imwrite(output_path, img)

            messagebox.showinfo("Success", 
                              f"Saved {len(self.processing_data['images'])} processed images!")
            self.processing_window.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Final processing failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()