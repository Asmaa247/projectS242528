from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3, os, datetime, uvicorn, smtplib
from email.mime.text import MIMEText
from openai import OpenAI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

# Set your OpenAI API key and other env variables
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Doctor Chatbot API")

# Allow CORS (for development, allowing all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE = "app.db"


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]      


def generate_doctor_response():
    
    qa_system_prompt = f"""
        You are a medical assistant. Provide health-related advice based on symptoms.Only answer questions related to health and medicine.If questions is not related to health and medication say i dont know.
        """


    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt), #qa_system_prompt
            MessagesPlaceholder(variable_name="input"),
        ]
    )

    chain = qa_prompt | llm


    conversational_rag_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
    )

    return conversational_rag_chain

def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    # Patients table
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT,
            dob TEXT
        )
    ''')
    # Doctors table
    c.execute('''
        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT,
            specialization TEXT
        )
    ''')
    # Records table with an added specialty column
    c.execute('''
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            symptoms TEXT,
            diagnosis TEXT,
            timestamp TEXT,
            doctor_id INTEGER,
            specialty TEXT,
            feedback TEXT,
            feedback_timestamp TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(id),
            FOREIGN KEY(doctor_id) REFERENCES doctors(id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def get_db_connection():
    conn = sqlite3.connect(DATABASE, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

# Dummy email sender (update SMTP settings in your .env)
def send_email(to_email: str, subject: str, body: str):
    smtp_server = os.getenv("EMAIL_HOST")
    smtp_port = int(os.getenv("EMAIL_PORT", "587"))
    smtp_user = os.getenv("EMAIL_HOST_USER")
    smtp_password = os.getenv("EMAIL_HOST_PASSWORD")
    if not smtp_server or not smtp_user:
        print("SMTP not configured")
        return
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = smtp_user
    msg['To'] = to_email

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)

# Pydantic models
class RegisterPatient(BaseModel):
    name: str
    email: str
    password: str
    dob: str  # YYYY-MM-DD

class RegisterDoctor(BaseModel):
    name: str
    email: str
    password: str
    specialization: str

class LoginData(BaseModel):
    email: str
    password: str
    role: str  # "patient" or "doctor"

# Updated FeedbackData with doctor_id
class FeedbackData(BaseModel):
    record_id: int
    feedback: str
    doctor_id: int

@app.post("/api/register/patient")
def register_patient(data: RegisterPatient):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO patients (name, email, password, dob) VALUES (?, ?, ?, ?)",
                  (data.name, data.email, data.password, data.dob))
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Email already registered")
    finally:
        conn.close()
    return {"message": "Patient registered successfully"}

@app.post("/api/register/doctor")
def register_doctor(data: RegisterDoctor):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO doctors (name, email, password, specialization) VALUES (?, ?, ?, ?)",
                  (data.name, data.email, data.password, data.specialization))
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Email already registered")
    finally:
        conn.close()
    return {"message": "Doctor registered successfully"}

@app.post("/api/login")
def login(data: LoginData):
    conn = get_db_connection()
    c = conn.cursor()
    table = "patients" if data.role == "patient" else "doctors"
    c.execute(f"SELECT * FROM {table} WHERE email=? AND password=?", (data.email, data.password))
    user = c.fetchone()
    conn.close()
    if user:
        return {"message": "Login successful", "user": dict(user)}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    contents = await file.read()
    tmp_filename = f"temp_{file.filename}"
    with open(tmp_filename, "wb") as f:
        f.write(contents)
    try:
        client = OpenAI()
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=open(tmp_filename, "rb")
        )
    except Exception as e:
        os.remove(tmp_filename)
        raise HTTPException(status_code=500, detail=str(e))
    os.remove(tmp_filename)
    # return {"transcription": transcription["text"]}
    print(transcription)
    return {"transcription": transcription.text}

@app.post("/api/diagnosis")
def get_diagnosis_api(symptoms: str = Form(...)):
    try:
        # response = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {"role": "system", "content": "You are a medical assistant. Provide health-related advice based on symptoms."},
        #         {"role": "user", "content": f"Patient symptoms: {symptoms}"}
        #     ]
        # )
        # diagnosis = response.choices[0].message["content"].strip()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        diagnosis = llm.invoke(f"You are a medical assistant. Provide health-related advice based on symptoms.\n Patient symptoms: {symptoms}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"diagnosis": diagnosis.content}

@app.post("/api/patient/submit")
async def patient_submit(symptoms: str = Form(...), diagnosis: str = Form(...),
                         patient_id: int = Form(...), specialty: str = Form(...),
                         audio: UploadFile = File(None)):
    conn = get_db_connection()
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO records (patient_id, symptoms, diagnosis, timestamp, specialty) VALUES (?, ?, ?, ?, ?)",
              (patient_id, symptoms, diagnosis, timestamp, specialty))
    conn.commit()
    record_id = c.lastrowid
    conn.close()
    return {"message": "Record submitted", "record_id": record_id}

@app.post("/api/patient/share")
def patient_share(record_id: int = Form(...), doctor_id: int = Form(...)):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE records SET doctor_id=? WHERE id=?", (doctor_id, record_id))
    conn.commit()
    conn.close()
    return {"message": "Record shared with doctor"}

@app.get("/api/doctor/records/{doctor_id}")
def doctor_records(doctor_id: int):
    conn = get_db_connection()
    c = conn.cursor()
    # Get doctor's specialization
    c.execute("SELECT * FROM doctors WHERE id=?", (doctor_id,))
    doctor = c.fetchone()
    if not doctor:
        conn.close()
        return {"records": []}
    specialty = doctor["specialization"]
    # Fetch records with matching specialty that have not received feedback (pending feedback)
    c.execute("""
        SELECT r.*, p.name as patient_name, p.email as patient_email 
        FROM records r 
        JOIN patients p ON r.patient_id = p.id 
        WHERE r.specialty = ? 
          AND (r.doctor_id IS NULL OR r.doctor_id = ?)
          AND r.feedback IS NULL
    """, (specialty, doctor_id))
    records = c.fetchall()
    conn.close()
    return {"records": [dict(record) for record in records]}


@app.post("/api/doctor/feedback")
def doctor_feedback(data: FeedbackData):
    conn = get_db_connection()
    c = conn.cursor()
    feedback_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Update the record with feedback, feedback_timestamp, and assign the doctor_id
    c.execute("UPDATE records SET feedback=?, feedback_timestamp=?, doctor_id=? WHERE id=?", 
              (data.feedback, feedback_timestamp, data.doctor_id, data.record_id))
    conn.commit()
    c.execute("SELECT p.email FROM records r JOIN patients p ON r.patient_id = p.id WHERE r.id=?", 
              (data.record_id,))
    row = c.fetchone()
    conn.close()
    if row:
        patient_email = row[0]
        send_email(patient_email, "Feedback on Your Diagnosis", 
                   f"A doctor has provided feedback on your submitted symptoms and diagnosis: {data.feedback}")
    return {"message": "Feedback submitted and patient notified"}

# New endpoint: Doctor History (only records with feedback)
@app.get("/api/doctor/history/{doctor_id}")
def doctor_history(doctor_id: int):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        SELECT r.id, r.symptoms, r.diagnosis, r.timestamp, r.feedback, r.feedback_timestamp,
               p.name as patient_name, p.email as patient_email
        FROM records r
        JOIN patients p ON r.patient_id = p.id
        WHERE r.doctor_id = ? AND r.feedback IS NOT NULL
        ORDER BY r.feedback_timestamp DESC
    """, (doctor_id,))
    records = c.fetchall()
    conn.close()
    return {"history": [dict(record) for record in records]}


# New endpoint: Patient History
@app.get("/api/patient/history/{patient_id}")
def patient_history(patient_id: int):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        SELECT r.id, r.symptoms, r.diagnosis, r.timestamp, r.feedback, r.feedback_timestamp,
               d.name as doctor_name, d.specialization as doctor_specialty
        FROM records r
        LEFT JOIN doctors d ON r.doctor_id = d.id
        WHERE r.patient_id = ?
        ORDER BY r.timestamp DESC
    """, (patient_id,))
    records = c.fetchall()
    conn.close()
    return {"history": [dict(record) for record in records]}

# Admin dashboard HTML template (password hidden)
ADMIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Admin Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #2c3e50; color: white; padding: 15px; border-radius: 5px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        tr:hover { background-color: #f5f5f5; }
        .table-container { margin: 25px 0; }
        h2 { color: #34495e; }
        .stats { display: flex; gap: 20px; margin: 20px 0; }
        .stat-box { 
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            min-width: 200px;
        }
        .password-placeholder {
            color: #999;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Admin Dashboard</h1>
    </div>
    
    <div class="stats">
        <div class="stat-box">
            <h3>📊 Statistics</h3>
            <p>Patients: {{ patients_count }}</p>
            <p>Doctors: {{ doctors_count }}</p>
            <p>Records: {{ records_count }}</p>
        </div>
    </div>

    {% for table in tables %}
    <div class="table-container">
        <h2>{{ table.name }} Table ({{ table.row_count }} rows)</h2>
        <table>
            <thead>
                <tr>
                    {% for column in table.filtered_columns %}
                    <th>{{ column }}</th>
                    {% endfor %}
                </tr>
            </thead>
            <tbody>
                {% for row in table.filtered_rows %}
                <tr>
                    {% for value in row %}
                    <td>
                        {% if column == 'password' %}
                            <span class="password-placeholder">••••••••</span>
                        {% else %}
                            {{ value }}
                        {% endif %}
                    </td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% endfor %}
</body>
</html>
"""

# Update the admin endpoint
@app.get("/admin", response_class=HTMLResponse)
def admin_panel():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Get counts for stats
    stats = {
        "patients_count": c.execute("SELECT COUNT(*) FROM patients").fetchone()[0],
        "doctors_count": c.execute("SELECT COUNT(*) FROM doctors").fetchone()[0],
        "records_count": c.execute("SELECT COUNT(*) FROM records").fetchone()[0],
    }
    
    # Get all table names
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [table[0] for table in c.fetchall()]
    
    # Prepare table data with password filtering
    table_data = []
    for table_name in tables:
        c.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in c.fetchall()]
        
        c.execute(f"SELECT * FROM {table_name}")
        rows = c.fetchall()
        
        # Filter password column
        filtered_columns = [col for col in columns if col.lower() != 'password']
        password_index = None
        if 'password' in columns:
            password_index = columns.index('password')
        
        # Filter rows
        filtered_rows = []
        for row in rows:
            row = list(row)
            if password_index is not None:
                del row[password_index]
            filtered_rows.append(tuple(row))
        
        table_data.append({
            "name": table_name,
            "filtered_columns": filtered_columns,
            "filtered_rows": filtered_rows,
            "row_count": len(rows)
        })
    
    conn.close()
    
    # Render template
    from jinja2 import Template
    template = Template(ADMIN_TEMPLATE)
    return template.render(tables=table_data, **stats)


# POST endpoint to handle chat queries
@app.post("/doctor-chat/")
async def handle_chat(chat_query:dict):
    
    conversational_rag_chain = generate_doctor_response()
    response = conversational_rag_chain.invoke(
        {"input": [HumanMessage(content=chat_query["query"])]},
        config={"configurable": {"session_id": chat_query["session_id"]}}
    )

    if response.content:
        return {"id":200 , "message": response.content , "type": "message" , "status":True}
    else:
        return {"id":200 , "message": "No answer found" , "type": "message" , "status":False}

def start_server():
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        log_level="debug",
        reload=True,
    )

if __name__ == "__main__":
    start_server()
