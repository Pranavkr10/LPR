create database lpr;
use lpr;
CREATE TABLE vehicle_registration (
    plate_no VARCHAR(20) PRIMARY KEY,
    owner_name VARCHAR(100),
    address TEXT,
    vehicle_class VARCHAR(50),
    fuel_type VARCHAR(20),
    engine_no VARCHAR(50),
    maker_model VARCHAR(50),
    colour VARCHAR(20),
    seating_capacity INT,
    insurance_date DATE,
    fitness_date DATE,
    registration_valid_date DATE,
    registration_authority VARCHAR(100),
    hypothecation VARCHAR(100)
);
 
INSERT INTO vehicle_registration (
    plate_no, owner_name, address, vehicle_class, fuel_type, engine_no, maker_model, colour,
    seating_capacity, insurance_date, fitness_date, registration_valid_date, registration_authority, hypothecation
) VALUES
    ('MH20EJ0364', 'Rahul Sharma', 'Pune, Maharashtra', 'D1 Segment SUV', 'Petrol', 'ENG1234567', 'Skoda', 'Blue',
     5, '2025-05-11', '2026-05-11', '2035-05-11', 'Pune RTO', 'HDFC Bank'),
    ('DL1CE1602', 'Ramesh Jatt', 'Delhi NCT, Delhi', 'LMV', 'Petrol', 'ENG3214567', 'Mahindra', 'Green',
     5, '2012-05-11', '2020-05-11', '2025-09-11', 'Delhi RTO', 'SBI Bank'),
    ('GJ04EE6437', 'Ambar Pratap', 'Surat, Gujarat', 'C-Segment', 'Petrol', 'ENG4567123', 'Hyundai', 'White',
     5, '2023-02-08', '2024-03-12', '2034-06-12', 'Gujarat RTO', 'ICIC Bank'),
    ('KL07CN7233', 'Venu Gopal Iyer', 'Munnar, Kerala', 'Mini MPV', 'Petrol', 'ENG9823567', 'Maruti Suzuki', 'White',
     5, '2011-11-05', '2020-11-01', '2025-11-11', 'Kerala RTO', 'CSB Bank'),
    ('UP85BW1000', 'Sunny Yadav', 'Gorakhpur, Uttar Pradesh', 'SUV', 'Petrol', 'ENG765123', 'Hyundai', 'Black',
     5, '2023-08-10', '2026-10-08', '2045-02-10', 'Uttar Pradesh RTO', 'BOI Bank'),
    ('HR26FC2782', 'Aman Batra', 'Sonipat, Haryana', 'Hybrid-MPV', 'Petrol', 'ENG098765', 'Hyundai', 'Blue',
     5, '2023-10-10', '2026-03-08', '2042-02-01', 'Haryana RTO', 'PNB Bank'),
    ('RJ45CR105', 'Jayesh Singh', 'Jaisalmer, Rajasthan', 'SUV', 'Petrol', 'ENG890765', 'Maserati', 'Grey',
     5, '2025-10-10', '2026-07-05', '2042-06-11', 'Rajasthan RTO', NULL);

     
delete from vehicle_registration where plate_no ='RJ45CR105';  
  
INSERT INTO vehicle_registration(plate_no, owner_name, address, vehicle_class, fuel_type, engine_no, maker_model, colour,
    seating_capacity, insurance_date, fitness_date, registration_valid_date, registration_authority, hypothecation)value(
    'RJ45CR5105', 'Jayesh Singh', 'Jaisalmer, Rajasthan', 'SUV', 'Petrol', 'ENG890765', 'Maserati', 'Grey',
     5, '2025-10-10', '2026-07-05', '2042-06-11', 'Rajasthan RTO', NULL);

SHOW tables;
SELECT *FROM VEHICLE_REGISTRATION;

INSERT INTO vehicle_registration(plate_no, owner_name, address, vehicle_class, fuel_type, engine_no, maker_model, colour,
    seating_capacity, insurance_date, fitness_date, registration_valid_date, registration_authority, hypothecation)value(
    'JH0TES8923', 'Kartki Toppo', 'Jamshedpur, Jharkhand', 'Sedan', 'Petrol', 'ENG543210', 'Maruti-Suzuki', 'White',
     5, '2019-10-10', '2027-05-02', '2037-02-03', 'Jharkhand RTO', 'Union Bank');