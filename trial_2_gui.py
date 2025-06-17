import customtkinter as ctk
from customtkinter import *
from tkinter import filedialog
from backend import plateLocalization
from PIL import Image
import mysql.connector
import re
import cv2 as cv

app = ctk.CTk()
app.geometry("900x900")  
set_appearance_mode("dark")

# Database connection
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password=" ",
    database="lpr"
)
cursor = conn.cursor()

def validatePlateNumber(plate):
    pattern = re.compile(r"^[A-Z]{2}\d[A-Z]{1,2}\d{1,4}$")
    return len(plate) == 10 and pattern.match(plate)

def getVehicleDetails(plate_num):
    query = "SELECT * FROM vehicle_registration WHERE plate_no = %s"
    cursor.execute(query, (plate_num,))
    return cursor.fetchone()

def getPlateNumsFromStr(text):
    #Extract all potential plate numbers from OCR text
    pattern = re.compile(r"[A-Z]{2}[0-9A-Z]{6,10}")  #Broader pattern to capture variations
    return pattern.findall(text)

def correctPlateNumber(plate):
    
    #Remove extra characters beyond 10 characters
    plate = plate[:10]
    
    #Convert to list for mutation
    plate_chars = list(plate)
    ###########################################################################################   
    '''
    Rule 1: First two characters should be alphabets (state code)
    Rule 2: Characters at index 2 and 3 should be digits 
    '''
    ##########################################################################################
    if len(plate_chars) > 3:
        if plate_chars[2] == 'O':
            plate_chars[2] = '0'
        elif plate_chars[3] == 'O':
            plate_chars[3] = '0'
        elif plate_chars[2] == 'Z':
            plate_chars[2] = '2'
        elif plate_chars[3] == 'Z':
            plate_chars[3] = '2'
        elif plate_chars[2] == 'B':
            plate_chars[2] == '8'
        elif plate_chars[3] == 'Z':
            plate_chars[3] = '2'
    ############################################################################################
    '''
    Rule 3: Last four characters should be digits
    Replace alphabet with numeric in the last four pos
    '''
    ###########################################################################################
    for i in range(max(0, len(plate_chars)-4), len(plate_chars)):
        if plate_chars[i] == 'O':
            plate_chars[i] = '0'
        elif plate_chars[i] == 'Z':
            plate_chars[i] = '2'
        elif plate_chars[i] == 'B':
            plate_chars[i] == '8'
    
    return ''.join(plate_chars)

def displayDetails(details):
    if details:
        details_text = f"""
        Plate Number: {details[0]}
        Owner: {details[1]}, Address: {details[2]}
        Class: {details[3]}, Fuel Type: {details[4]}
        Engine Number: {details[5]}
        Vehicle: {details[6]}, Colour: {details[7]}
        Seating Capacity: {details[8]}, Insurance Date Upto: {details[9]}
        Fitness Date Upto: {details[10]},
        Registration Valid Until: {details[11]}
        Registration Authority: {details[12]}, Hypothecation: {details[13]}"""
        details_label.configure(text=details_text)
    else:
        details_label.configure(text="No record found")


def BasicGUI():
    imgPath = filedialog.askopenfilename(title="Select image", filetypes=[("Images", "*.png *.jpg *.jpeg")])
    if not imgPath: 
        return
    
    
    processed_img, ocr_text = plateLocalization(imgPath)
    displayImg(processed_img)
    
    
    plate_nums = getPlateNumsFromStr(ocr_text)
    print("Raw OCR plates detected:", plate_nums)
    
    #Apply corrections to all detected plates
    corrected_plates = [correctPlateNumber(plate) for plate in plate_nums]
    print("Corrected plates:", corrected_plates)
    
    #Trying  each corrected plate until we find a match
    found = False
    for plate in corrected_plates:
        vehicle_details = getVehicleDetails(plate)
        if vehicle_details:
            displayDetails(vehicle_details)
            found = True
            break
    
    if not found:
        details_label.configure(text=f"No records found for plates: {', '.join(corrected_plates)}")
        
def displayImg(img_arr):
    img_rgb = cv.cvtColor(img_arr, cv.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((650, 650))
    img_tk = ctk.CTkImage(light_image=img_pil, size=(650, 650))
    image_label.configure(image=img_tk)
    image_label.image = img_tk


btn = ctk.CTkButton(master=app, text="Select image", corner_radius=32, fg_color="#69229B", hover_color="#4158D0", border_color="#031C5B", border_width=2, command=BasicGUI)

btn.pack(padx=30, pady=20)
image_label = ctk.CTkLabel(app, text="")
image_label.pack()
details_label = ctk.CTkLabel(app, text="Vehicle details will appear here", wraplength=400, justify="left")
details_label.pack(pady=10)
app.mainloop()