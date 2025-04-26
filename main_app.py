from flask import Flask, request, render_template, redirect, url_for, flash, jsonify,session
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os
import numpy as np
import datetime
import time
import serial
import smtplib
from email.mime.text import MIMEText

from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.exc import IntegrityError
from collections import deque
import asyncio
import threading
import time
import numpy as np
from collections import deque
from bleak import BleakClient
from email_send import send_email
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from flask_socketio import SocketIO, emit
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch

# Hugging Face Authentication
HF_token = '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'
login(HF_token)

# Load model
model_name = "ruslanmv/ai-medical-model-32bit"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=HF_token)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, token=HF_token)
model = model.to(device)


load_dotenv()

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key')
socketio = SocketIO(app, cors_allowed_origins="*")
# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Redirects to this view if login is required

# Set your OpenAI API key and other env variables
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

DEVICE_ADDRESS = "F9:DF:AD:F5:7D:4E"
HRM_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

SERIAL_PORT = "COM8"  # Update for Linux/Mac: "/dev/ttyUSB0"
BAUD_RATE = 9600
WINDOW_SIZE = 10  # Rolling average window for smoothing
LOW_HR = 50       # BPM threshold for low heart rate
HIGH_HR = 100     # BPM threshold for high heart rate
MAX_HISTORY = 100  # Max data points for the graph


ble_results = {}
active_threads = {}
# Database Models
class Patient(db.Model):
    __tablename__ = 'patients'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    dob = db.Column(db.String(200))
    records = db.relationship('Record', backref='patient', lazy=True)

class Doctor(db.Model):
    __tablename__ = 'doctors'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    specialization = db.Column(db.String(100), nullable=False)
    records = db.relationship('Record', backref='doctor', lazy=True)

class Record(db.Model):
    __tablename__ = 'records'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    symptoms = db.Column(db.Text, nullable=True)
    diagnosis = db.Column(db.Text, nullable=True)
    specialty = db.Column(db.String(100), nullable=True)
    timestamp = db.Column(db.String(20), nullable=True)
    nationality = db.Column(db.String(100))
    chronic_illnesses = db.Column(db.Text)
    medications = db.Column(db.Text)
    surgeries = db.Column(db.Text)
    allergies = db.Column(db.Text)
    heart_rate = db.Column(db.String(255))
    family_history = db.Column(db.Text)
    height = db.Column(db.String(255))
    weight = db.Column(db.String(255)) 
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'))
    feedback = db.Column(db.Text)
    feedback_timestamp = db.Column(db.String(20))

heart_user_id = ""
# Initialize database
with app.app_context():
    db.create_all()

class UserWrapper(UserMixin):
    def __init__(self, user_obj, role):
        self.user_obj = user_obj
        self.role = role
        self.id = f"{role}:{user_obj.id}"  # Unique composite ID

    def get_id(self):
        return self.id

    def get_user(self):
        return self.user_obj

    def get_role(self):
        return self.role

    @property
    def is_authenticated(self):
        return True  # Always authenticated if wrapper is valid

    @property
    def is_active(self):
        return True  # Modify if you have active/inactive status

    @property
    def is_anonymous(self):
        return False  # Not anonymous since user is wrapped



@login_manager.user_loader
def load_user(user_id):
    try:
        
        print("_____________________****cur*********************_____________________role")
        print(user_id)
        role, obj_id = user_id.split(":")
        if role == 'patient':
            user = Patient.query.get(int(obj_id))
        elif role == 'doctor':
            user = Doctor.query.get(int(obj_id))
        else:
            return None

        return UserWrapper(user, role) if user else None
    except Exception:
        return None


# Chatbot setup

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]      



def get_diagnosiss(symptoms):
    try:
        question = f"The patient reports the following symptoms: {symptoms}. What could be the diagnosis?"
        prompt = f"System: You are a Medical AI assistant. {question}"
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=500)
        diagnosis = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return diagnosis
    except Exception as e:
        return f"Error generating diagnosis: {e}"












@app.route("/")
@login_required
def index():
    role, patient_id = current_user.id.split(":")
    stop_thread(patient_id)
    return render_template('index.html')

@app.route('/api/diagnosis', methods=['POST'])
def get_diagnosis():
    symptoms = request.form.get('symptoms') 
    print(symptoms)
    # symptoms = data.get('symptoms')
    diagnosis = get_diagnosiss(symptoms)
    
    print(diagnosis)
    return jsonify({'diagnosis': diagnosis})

# Routes that need templates
@app.route('/register/patient', methods=['GET', 'POST'])
def register_patient():
    if current_user.is_authenticated:
        try:
            role, patient_id = current_user.id.split(":")
            stop_thread(patient_id)  # Only stop thread if logged in
        except Exception as e:
            print("⚠️ Error stopping thread:", e)
    
    if request.method == 'POST':
        try:
            hashed_password = generate_password_hash(request.form['password'])
            patient = Patient(
                name=request.form['name'],
                email=request.form['email'],
                password=hashed_password,
                dob=request.form['dob']
            )
            db.session.add(patient)
            db.session.commit()
            flash('Patient registered successfully', 'success')
            return redirect(url_for('login'))
        except IntegrityError:
            db.session.rollback()
            flash('Email already registered', 'error')
        except Exception as e:
            db.session.rollback()
            flash(str(e), 'error')

    return render_template('register_patient.html')

@app.route('/register/doctor', methods=['GET', 'POST'])
def register_doctor():
    
    if request.method == 'POST':
        try:
            hashed_password = generate_password_hash(request.form['password'])
            doctor = Doctor(
                name=request.form['name'],
                email=request.form['email'],
                password=hashed_password,
                specialization=request.form['specialization']
            )
            db.session.add(doctor)
            db.session.commit()
            flash('Doctor registered successfully', 'success')
            return redirect(url_for('login'))
        except IntegrityError:
            db.session.rollback()
            flash('Email already registered', 'error')
        except Exception as e:
            db.session.rollback()
            flash(str(e), 'error')
    
    return render_template('register_doctor.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        role = request.form.get('role')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if role == 'patient':
            user = Patient.query.filter_by(email=email).first()
            dashboard_route = 'patient_dashboard'
        elif role == 'doctor':
            user = Doctor.query.filter_by(email=email).first()
            dashboard_route = 'doctor_dashboard'
        else:
            flash('Invalid role', 'error')
            return redirect(url_for('login'))
        
        if user and check_password_hash(user.password, password):
            # Store user in session (you'll need to implement proper session management)
            login_user(UserWrapper(user, role))
            flash('Login successful', 'success')
            return redirect(url_for(dashboard_route, user_id=user.id))
        else:
            flash('Invalid credentials', 'error')
    
    return render_template('login.html')

@app.route('/patient/dashboard/<int:user_id>')
@login_required
def patient_dashboard(user_id):
    role, patient_id = current_user.id.split(":")
    stop_thread(patient_id)
    patient = Patient.query.get_or_404(user_id)
    records = Record.query.filter_by(patient_id=user_id).order_by(Record.timestamp.desc()).all()
    return render_template('dashboard_patient.html', patient=patient, records=records)

@app.route('/doctor/dashboard/<int:user_id>')
def doctor_dashboard(user_id):
    doctor = Doctor.query.get_or_404(user_id)
    # Get pending records for doctor's specialty
    records = db.session.query(Record, Patient.name, Patient.dob).join(
        Patient, Record.patient_id == Patient.id
    ).filter(
        Record.specialty == doctor.specialization,
        db.or_(
            Record.doctor_id.is_(None),
            Record.doctor_id == doctor.id
        ),
        Record.feedback.is_(None)
    ).all()
    
    # Get doctor's feedback history
    history = db.session.query(
        Record, Patient.name
    ).join(
        Patient, Record.patient_id == Patient.id
    ).filter(
        Record.doctor_id == doctor.id,
        Record.feedback.isnot(None)
    ).order_by(
        Record.feedback_timestamp.desc()
    ).all()
    
    return render_template('dashboard_doctor.html', doctor=doctor, records=records, history=history)

@app.route('/doctor/edit_record/<int:record_id>/<action>', methods=['POST'])
@login_required
def edit_record(record_id, action):
    record = Record.query.get_or_404(record_id)

    record.diagnosis = request.form.get('diagnosis')
    record.medications = request.form.get('medications')
    record.allergies = request.form.get('allergies')
    record.chronic_illnesses = request.form.get('chronic_illnesses')
    record.surgeries = request.form.get('surgeries')

    db.session.commit()
    flash('Record updated successfully!', 'success')
    role, id = current_user.id.split(":")
    
    if action == "patient":
        return redirect(url_for('patient_dashboard', user_id= id))
    
    return redirect(url_for('doctor_dashboard', user_id=id))

@app.route('/patient/submit-record/<int:user_id>', methods=['GET', 'POST'])
def submit_record(user_id):
    if request.method == 'POST':
        
        print(request.form)
        
        try:
            nationality = request.form.get('nationality')
            medications = request.form.get('medications')
            surgeries = request.form.get('surgeries')
            allergies = request.form.get('allergies')
            specialty = request.form.get('specialty')
            height = request.form.get('height')
            weight = request.form.get('weight')

            # Multi-value fields (checkboxes)
            chronic_illnesses = request.form.getlist('chronic_illnesses')  # May have multiple or just one
            family_history = request.form.getlist('family_history')

            # Optional: join them as strings if storing in DB as one field
            chronic_illnesses_str = ", ".join(chronic_illnesses)
            family_history_str = ", ".join(family_history)
            
           
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            
            record = Record(
                patient_id=user_id,
                height = height,
                weight = weight,
                timestamp=timestamp,
                specialty=specialty,
                nationality = nationality,
                chronic_illnesses = chronic_illnesses_str,
                medications = medications,
                surgeries = surgeries,
                allergies = allergies,
                family_history = family_history_str,
            )
            db.session.add(record)
            db.session.commit()
          
          
  
            flash('Record submitted successfully', 'success')
            return redirect(url_for('patient_dashboard', user_id=user_id))
        except Exception as e:
            db.session.rollback()
            flash(str(e), 'error')
    
    return redirect(url_for('patient_dashboard', user_id=user_id))

@app.route('/doctor/provide-feedback/<int:record_id>/<int:doctor_id>', methods=['GET', 'POST'])
def provide_feedback(record_id, doctor_id):
    record = Record.query.get_or_404(record_id)
    doctor = Doctor.query.get_or_404(doctor_id)
    
    if request.method == 'POST':
        try:
            feedback = request.form.get('feedback')
            if not feedback:
                flash('Please provide feedback', 'error')
                return redirect(url_for('provide_feedback', record_id=record_id, doctor_id=doctor_id))
            
            record.feedback = feedback
            record.feedback_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record.doctor_id = doctor_id
            db.session.commit()
            
            # Send email to patient
            patient = Patient.query.get(record.patient_id)
            send_email("Doctor Feedback", patient.email, patient.name, "feedback", patient_name=None, symptoms=feedback, doctor_name=None)
            # if patient and patient.email:
            #     send_email(
            #         patient.email,
            #         "Feedback on Your Diagnosis",
            #         f"A doctor has provided feedback on your submitted symptoms and diagnosis: {feedback}"
            #     )
            
            flash('Feedback submitted successfully', 'success')
            return redirect(url_for('doctor_dashboard', user_id=doctor_id))
        except Exception as e:
            
            db.session.rollback()
            flash(str(e), 'error')
    
    return redirect(url_for('doctor_dashboard', user_id= doctor_id))

@app.route('/chat/<int:user_id>', methods=['GET', 'POST'])
def chat(user_id):
    if request.method == 'POST':
        try:
            query = request.form.get('query')
            session_id = f"user_{user_id}"  # Using user_id as session_id
            
            conversational_rag_chain = generate_doctor_response()
            response = conversational_rag_chain.invoke(
                {"input": [HumanMessage(content=query)]},
                config={"configurable": {"session_id": session_id}}
            )
            
            if response.content:
                flash(response.content, 'chat')
            else:
                flash("No answer found", 'chat-error')
            
            return redirect(url_for('chat', user_id=user_id))
        except Exception as e:
            flash(str(e), 'error')
    
    return render_template('chat.html', user_id=user_id)




@app.route("/logout")
@login_required
def logout():
    role, patient_id = current_user.id.split(":")
    stop_thread(patient_id)
    logout_user()
    flash("You have been logged out successfully.", "info")  # Flash message
    return redirect(url_for("login"))  # Redirect to login page





@app.route("/save_heart_rate", methods=['POST'])
@login_required
def save_heart_rate():
    bpm = request.form['bpm']
    symptoms = request.form['symptoms']
    diagnosis = request.form['diagnosis']
    
    print(current_user.id)
    print(current_user.id.split(':'))
    role, id = current_user.id.split(":")
    record = Record.query.filter_by(patient_id = id).first()
    
    if record:
        
        doctors = Doctor.query.filter_by(specialization = record.specialty).all()
            
        for doctor in doctors:
            print("here")
            send_email("New Patient Application for Review", doctor.email, "", "doctor", "",symptoms, "doctor abc")
        record.heart_rate = bpm
        record.symptoms = symptoms
        record.diagnosis = diagnosis
        
        db.session.commit()
    else:
        pass
    
    return redirect(url_for('patient_dashboard', user_id= id))
    

@app.route('/heart_rate')
@login_required
def heart_rate():
    role, patient_id = current_user.id.split(":")
    stop_thread(patient_id)
    global heart_user_id
    role, user_id = current_user.id.split(":")
    heart_user_id = user_id
    return render_template('heart_rate.html', patient_id=user_id)
 
 
def connect_serial():
    """Establishes serial connection and stores it in session."""
   
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(3)  # Allow Arduino to initialize
        return ser
    except serial.SerialException as e:
        return None

def handle_heart_rate(sender, data):
    
    
    print('result')
    if len(data) >= 2:
        bpm = int(data[1])
        if heart_user_id not in ble_results:
            ble_results[heart_user_id] = {
                'bpm_history': deque(maxlen=MAX_HISTORY),
                'time_history': deque(maxlen=MAX_HISTORY),
                'bpm_buffer': deque(maxlen=WINDOW_SIZE)
            }
            
        user_data = ble_results[heart_user_id]
        user_data['bpm_buffer'].append(bpm)
        avg_bpm = int(np.mean(user_data['bpm_buffer']))

        user_data['bpm_history'].append(avg_bpm)
        user_data['time_history'].append(time.strftime("%H:%M:%S"))
        
        ble_results[heart_user_id] = {
            'bpm': avg_bpm,
            'status': (
                "Low Heart Rate" if avg_bpm < LOW_HR else
                "High Heart Rate" if avg_bpm > HIGH_HR else
                "Normal Heart Rate"
            ),
            'bpm_history': list(user_data['bpm_history']),
            'time_history': list(user_data['time_history']),
            'bpm_buffer': user_data['bpm_buffer']
        }
        
        
        
        result = ble_results.get(heart_user_id)
        if result:
            print('_________________')
            print(result)
            socketio.emit(f"heart_rate_update_{heart_user_id}", {
                'bpm': result['bpm'],
                'status': result['status'],
                'bpm_history': result['bpm_history'],
                'time_history': result['time_history'],
            })
        time.sleep(1)

import random
async def simulate_heart_rate_data(stop_event):
    
    sender = None
    print(stop_event.is_set())
    while not stop_event.is_set():
        # Generate random BPM between 50 and 120
        fake_bpm = random.randint(50, 120)
        data = [0, fake_bpm]  # Simulating [header, bpm]
        
        handle_heart_rate(sender, data)
        time.sleep(1) 

async def connect_and_read(stop_event):
    while not stop_event.is_set():  # Keep trying to reconnect
        try:
            async with BleakClient(DEVICE_ADDRESS) as client:
                if not await client.is_connected():
                    print("❌ Failed to connect.")
                    await asyncio.sleep(1)
                    continue

                await client.start_notify(HRM_CHAR_UUID, handle_heart_rate)
                try:
                    while True:
                        await asyncio.sleep(1)  # Keeps loop alive to process notifications
                except asyncio.CancelledError:
                    print("❌ BLE loop was cancelled")
                    break
                except Exception as e:
                    print(f"❌ Error in BLE connection: {e}")
                finally:
                    await client.stop_notify(HRM_CHAR_UUID)
        except Exception as e:
            print(f"❌ BLE connection error: {e}, retrying...")
            await asyncio.sleep(1)
        
        
def run_ble(stop_event):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(connect_and_read(stop_event))
    finally:
        loop.close()
    

def bluetooth_worker(user_id, stop_event):
 
    run_ble(stop_event)
    # This now runs continuously    
    time.sleep(1)
    
    
    # This part is not needed anymore as we emit from handle_heart_rate
    # while True:
    #     result = ble_results.get(user_id)
    #     if result:
    #         socketio.emit(f"heart_rate_update_{patient_id}", {
    #             'bpm': result['bpm'],
    #             'status': result['status'],
    #             'bpm_history': result['bpm_history'],
    #             'time_history': result['time_history'],
    #         })
    #     time.sleep(1)
        


import random

def test_sensor_worker(patient_id, stop_event):
    bpm_history = deque(maxlen=MAX_HISTORY)
    time_history = deque(maxlen=MAX_HISTORY)
    bpm_buffer = deque(maxlen=WINDOW_SIZE)

    while not stop_event.is_set():
        try:
            # Simulate a random heart rate between 50 and 120
            bpm_value = random.randint(50, 150)
            bpm_buffer.append(bpm_value)
            avg_bpm = int(np.mean(bpm_buffer))

            bpm_history.append(avg_bpm)
            time_history.append(time.strftime("%H:%M:%S"))

            if avg_bpm < LOW_HR:
                status = "Low Heart Rate"
            elif avg_bpm > HIGH_HR:
                status = "High Heart Rate"
            else:
                status = "Normal Heart Rate"

            # Save to DB
            # data = Record.query.filter_by(patient_id=patient_id).first()
            # if data:
            #     data.heart_rate = avg_bpm
            #     db.session.commit()

            # Emit update to frontend
            socketio.emit(f"heart_rate_update_{patient_id}", {
                'bpm': avg_bpm,
                'status': status,
                'bpm_history': list(bpm_history),
                'time_history': list(time_history)
            })

            time.sleep(1)

        except Exception as e:
            print(f"❌ Error in test sensor thread: {e}")
            time.sleep(1)

def sensor_worker(patient_id, stop_event):
    while not stop_event.is_set():
        try:
            ser = connect_serial()
            if not ser:
                print("❌ Failed to connect to Arduino")
                return

            bpm_history = deque(maxlen=MAX_HISTORY)
            time_history = deque(maxlen=MAX_HISTORY)
            bpm_buffer = deque(maxlen=WINDOW_SIZE)

            while not stop_event.is_set():
                line = ser.readline().decode().strip()
                if line.startswith("Heart Rate (BPM):"):
                    bpm_value = int(line.split(":")[-1].strip())
                    bpm_buffer.append(bpm_value)
                    avg_bpm = int(np.mean(bpm_buffer))

                    bpm_history.append(avg_bpm)
                    time_history.append(time.strftime("%H:%M:%S"))

                    if avg_bpm < LOW_HR:
                        status = "Low Heart Rate"
                    elif avg_bpm > HIGH_HR:
                        status = "High Heart Rate"
                    else:
                        status = "Normal Heart Rate"

                    # # Save to DB
                    # data = Record.query.filter_by(patient_id=patient_id).first()
                    # data.heart_rate = avg_bpm
                    # db.session.commit()

                    # Emit update to frontend
                    socketio.emit(f"heart_rate_update_{patient_id}", {
                        'bpm': avg_bpm,
                        'status': status,
                        'bpm_history': list(bpm_history),
                        'time_history': list(time_history)
                    })

                time.sleep(1)
        except Exception as e:
            print(f"❌ Error in sensor thread: {e}")   
        
    
@app.route('/heart_rate_monitor')
@login_required
def heart_rate_monitor():
    method = request.args.get('method')  # bluetooth or sensor
    role, patient_id = current_user.id.split(":")

    if patient_id in active_threads:
        return render_template('heart_rate_monitor.html', patient_id=patient_id)

    stop_event = threading.Event()

    if method == 'bluetooth':
        thread = threading.Thread(target=bluetooth_worker, args=(patient_id, stop_event))
    elif method == 'sensor':
        thread = threading.Thread(target=sensor_worker, args=(patient_id, stop_event))
    else:
        return "❌ Invalid method", 400

    thread.daemon = True
    thread.start()
    active_threads[patient_id] = (thread, stop_event)

    return render_template('heart_rate_monitor.html', patient_id=patient_id)



@app.route('/admin')
@login_required
def admin_dashboard():
    
    role, patient_id = current_user.id.split(":")
    stop_thread(patient_id)
    patients = Patient.query.all()
    doctors = Doctor.query.all()
    records = Record.query.all()

    return render_template('admin_dashboard.html', patients=patients, doctors=doctors, records=records)

def stop_thread(patient_id):
    try:
        if patient_id in active_threads:
            print(f"[Stop Thread] Stopping thread for {patient_id}")
            thread, stop_event = active_threads[patient_id]
            stop_event.set()
            thread.join(timeout=1)
            del active_threads[patient_id]
    except: 
        pass

# @app.before_request
# def stop_thread_if_needed():
#     try:
#         if request.endpoint != 'heart_rate_monitor':
            
#     except Exception as e:
#         print(f"[Stop Thread] Exception: {e}")

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)

