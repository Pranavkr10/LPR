from flask import Flask, render_template, request
from backend import plateLocalization
import mysql.connector
import cv2 as cv
import re
import os
import uuid

app=Flask(__name__)

###HELPER FUNCTIONS FOR PREPROCESSING THE OCR STRING###
def ValidPlateNumber(plate):
    pattern= re.compile(r"^[A-Z]{2}\d[A-Z]{1,2}\d{1,4}$")
    return len(plate) == 10 and pattern.match(plate)

def getVehicleDetails(cursor, plate_num):
    query= "SELECT * FROM vehicle_registration WHERE plate_no = %s"
    cursor.execute(query, (plate_num,))
    return cursor.fetchone()

def getVehicleNumFromStr(text):
    #extracting all potential plate numbers from the ocr
    pattern=re.compile(r"(?=([0-9A-Z]{8,10}))")
    return pattern.findall(text)

def correctPlateNumber(plate):
   plate = plate.replace(" ", "")
   n = len(plate)

   if n < 4:
       return plate
   
   plate_chars = list(plate)
   ALPHABET_ERROR_MAP = {'0': 'O', '1':  'I', '2': 'Z', '4': 'A', '5': 'S', '8': 'B'}
   DIGIT_ERROR_MAP = {'O':'0','I':'1','Z':'2','B':'8','S':'5','A':'4','D':'0','T':'1',
    'Y':'4','U':'4','L':'1','Q':'0','G':'6','J':'1','W':'8','H':'8','E':'8'}
   
   for i in range(0, 2):
       if plate_chars[i] in ALPHABET_ERROR_MAP:
           plate_chars[i] = ALPHABET_ERROR_MAP[plate_chars[i]]

   for i in range(2, min(4, n)):
        if plate_chars[i] in DIGIT_ERROR_MAP:
            plate_chars[i] = DIGIT_ERROR_MAP[plate_chars[i]]
        if not plate_chars[i].isdigit():
            plate_chars[i] = '0'

   for i in range(max(0, n-4), n):
        if plate_chars[i] in DIGIT_ERROR_MAP:
            plate_chars[i] = DIGIT_ERROR_MAP[plate_chars[i]]
        #if not plate_chars[i].isdigit():
            #plate_chars[i] = '0'

   return ''.join(plate_chars)

   
def displayDetails(details):
    if details:
        return f"""
       <table class="detail">
       <tr><th>Plate Number</th><td>{details[0]}</td></tr>
       <tr><th>Owner</th><td>{details[1]}</td></tr>
       <tr><th>Address</th><td>{details[2]}</td></tr>
       <tr><th>Class</th><td>{details[3]}</td></tr>
       <tr><th>Fuel Type</th><td>{details[4]}</td></tr>
       <tr><th>Engine Number</th><td>{details[5]}</td></tr>
       <tr><th>Vehicle</th><td>{details[6]}</td></tr>
       <tr><th>Colour</th><td>{details[7]}</td></tr>
       <tr><th>Seating Capacity</th><td>{details[8]}</td></tr>
       <tr><th>Insurance Date Upto</th><td>{details[9]}</td></tr>
       <tr><th>Fitness Upto</th><td>{details[10]}</td></tr>
       <tr><th>Registration Valid Upto</th><td>{details[11]}</td></tr>
       <tr><th>Regustration Authority</th><td>{details[12]}</td></tr>
       <tr><th>Hypothecation</th><td>{details[13]}</td></tr>
       <tr><th>Category</th><td>{details[14]}</td></tr>
        """
    else:
        return "No record found"

######ROUTES######
@app.route('/vehicle_tracking', methods=['GET', 'POST'])
def vehicle_tracking():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="qwerty",
        database="lpr"
    )
    #why dictionary=True? To get the result as a dictionary
    cursor = conn.cursor(dictionary=True)
    search_results = []
    if request.method == 'POST':
        last4Digits = request.form.get('last4Digits', '').strip()
        vehicle_colour = request.form.get('vehicle-colour', '').strip()
        if last4Digits and not last4Digits.isdigit():
            return "Invalid plate number format", 400 #why 400? Coz it is a client error
        if len(last4Digits) != 4:
            return "Plate number must conatin 4 digits", 400
        
        query=''' SELECT vt.plate_no, vt.owner_name, vt.colour, vt.phone_number, vt.bank_account_number, vt.social_media_handle, vt.vehicle_image
                FROM vehicle_tracking vt
                JOIN vehicle_registration vr on vt.plate_no = vr.plate_no
                WHERE 1=1
            '''
        params = []
        if last4Digits:
            query += "AND vt.plate_no LIKE %s"
            params.append(f"%{last4Digits}")

        if vehicle_colour:
            query += "AND LOWER(vr.colour) = %s"
            params.append(vehicle_colour)
    
        if params:
            cursor.execute(query, tuple(params))
            search_results = cursor.fetchall()

    cursor.close()
    conn.close()
    return render_template('vehicle_tracking.html', search_results=search_results)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No file uploaded', 400
        
        file = request.files['image']
        if file.filename == '':
            return 'No file selected', 400

        upload_filename = f"upload_{uuid.uuid4().hex[:8]}.png"
        dirPath = os.path.join('static', upload_filename)
        file.save(dirPath)
        
        try:
            processed_img, all_plate_texts = plateLocalization(dirPath)
        except Exception as e:
            return f"Processing error: {str(e)}", 500

        processed_filename = f"processed_{uuid.uuid4().hex[:8]}.png"
        processed_path = os.path.join('static', processed_filename)
        cv.imwrite(processed_path, processed_img)

        #Extracting all potential plate numbers 
        plate_nums = []
        for text in all_plate_texts:
            plate_nums.extend(getVehicleNumFromStr(text.upper()))
        
        #original + corrected versions
        all_candidates = []
        for plate in set(plate_nums):
            all_candidates.append(plate)

            corrected = correctPlateNumber(plate)
            if corrected != plate:
                all_candidates.append(corrected)
        
        all_candidates = list(set(all_candidates))
        
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="qwerty",
            database="lpr"
        )
        cursor = conn.cursor()
        
        vehicles = []
        all_criminal_records = []
        
        for plate in all_candidates:
            details = getVehicleDetails(cursor, plate)
            if details:
                cursor.execute('''SELECT plate_no, criminal_background FROM past_records 
                               WHERE plate_no = %s AND criminal_background != 'None' ''', (plate,))
                criminal_records = cursor.fetchall()
                
                vehicles.append({
                    'image': processed_filename,
                    'number': plate,
                    'details': displayDetails(details),
                    'category': details[14],
                    'criminal_records': criminal_records
                })
                
                if criminal_records:
                    all_criminal_records.extend(criminal_records)
        
        if not vehicles:
            vehicles.append({
                'image': processed_filename,
                'details': 'No matching record found'
            })
        
        cursor.close()
        conn.close()
        return render_template('index.html', 
                               vehicles=vehicles, 
                               all_criminal_records=all_criminal_records)
    
    return render_template('index.html', vehicles=[])

if __name__=='__main__':
    app.run(debug=True)