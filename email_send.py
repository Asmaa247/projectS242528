import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

APP_PASS = 'xwuq beqd rgym wlir'  # Your app password
SENDER_EMAIL = 'mairajuble@gmail.com'  # Your sender email

def send_email(subject, to_email, recipient_name, recipient_type, patient_name=None, symptoms=None, doctor_name=None):
    from_email = SENDER_EMAIL
    app_password = APP_PASS  # Replace with your actual app password

    if recipient_type == "patient":
        # Email content for the patient
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 0; padding: 0;">
            <table align="center" border="0" cellpadding="0" cellspacing="0" width="600">
                <tr>
                    <td align="center" bgcolor="#4CAF50" style="padding: 40px 0 30px 0; color: white; font-size: 24px;">
                        <strong>Application Received</strong>
                    </td>
                </tr>
                <tr>
                    <td bgcolor="#ffffff" style="padding: 40px 30px 40px 30px;">
                        <table border="0" cellpadding="0" cellspacing="0" width="100%">
                            <tr>
                                <td style="color: #153643; font-size: 24px;">
                                    <b>Dear {patient_name},</b>
                                </td>
                            </tr>
                            <tr>
                                <td style="padding: 20px 0 30px 0; color: #153643; font-size: 16px; line-height: 20px;">
                                    Thank you for submitting your information and symptoms. Your application has been received, and a specialized doctor will review it shortly. We will contact you soon with the next steps.
                                </td>
                            </tr>
                            <tr>
                                <td style="padding: 20px 0 30px 0; color: #153643; font-size: 16px; line-height: 20px;">
                                    Best regards, <br> Tabib Team
                                </td>
                            </tr>
                        </table>
                    </td>
                </tr>
                <tr>
                    <td bgcolor="#4CAF50" style="padding: 30px 30px 30px 30px;">
                        <table border="0" cellpadding="0" cellspacing="0" width="100%">
                            <tr>
                                <td style="color: white; font-size: 14px;" align="center">
                                    &copy; 2024 Tabib. All rights reserved.
                                </td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </body>
        </html>
        """
    elif recipient_type == "doctor":
        # Email content for the doctor
       html_content = f"""
<html>
<body style="font-family: Arial, sans-serif; margin: 0; padding: 0;">
    <table align="center" border="0" cellpadding="0" cellspacing="0" width="600">
        <tr>
            <td align="center" bgcolor="#4CAF50" style="padding: 40px 0 30px 0; color: white; font-size: 24px;">
                <strong>New Patient Application</strong>
            </td>
        </tr>
        <tr>
            <td bgcolor="#ffffff" style="padding: 40px 30px 40px 30px;">
                <table border="0" cellpadding="0" cellspacing="0" width="100%">
                    <tr>
                        <td style="color: #153643; font-size: 24px;">
                            <b>Dear {doctor_name},</b>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 20px 0 30px 0; color: #153643; font-size: 16px; line-height: 20px;">
                            A new patient application has been submitted. Please review the details and symptoms at your earliest convenience.
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 20px 0 30px 0; color: #153643; font-size: 16px; line-height: 20px;">
                            <strong>Patient's Name:</strong> {patient_name}<br>
                            <strong>Symptoms:</strong> {symptoms}<br>

                        </td>
                    </tr>
                  
                
                  
                    <tr>
                        <td style="padding: 20px 0 30px 0; color: #153643; font-size: 16px; line-height: 20px;">
                            Thank you,<br>Tabib. Team
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
        <tr>
            <td bgcolor="#4CAF50" style="padding: 30px 30px 30px 30px;">
                <table border="0" cellpadding="0" cellspacing="0" width="100%">
                    <tr>
                        <td style="color: white; font-size: 14px;" align="center">
                            &copy; 2024 tabib. All rights reserved.
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""

    elif recipient_type == "feedback":
            # Email content for the doctor
        html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; margin: 0; padding: 0;">
        <table align="center" border="0" cellpadding="0" cellspacing="0" width="600">
            <tr>
                <td align="center" bgcolor="#4CAF50" style="padding: 40px 0 30px 0; color: white; font-size: 24px;">
                    <strong>Doctor Feedback</strong>
                </td>
            </tr>
            <tr>
                <td bgcolor="#ffffff" style="padding: 40px 30px 40px 30px;">
                    <table border="0" cellpadding="0" cellspacing="0" width="100%">
                        <tr>
                            <td style="color: #153643; font-size: 24px;">
                                <b>Dear patient. {patient_name},</b>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 20px 0 30px 0; color: #153643; font-size: 16px; line-height: 20px;">
                                Feedback from doctor
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 20px 0 30px 0; color: #153643; font-size: 16px; line-height: 20px;">
                                <strong>doctor Name:</strong> {doctor_name}<br>
                                <strong>feedback:</strong> {symptoms}<br>

                            </td>
                        </tr>
                    
                    
                    
                        <tr>
                            <td style="padding: 20px 0 30px 0; color: #153643; font-size: 16px; line-height: 20px;">
                                Thank you,<br>Tabib. Team
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
            <tr>
                <td bgcolor="#4CAF50" style="padding: 30px 30px 30px 30px;">
                    <table border="0" cellpadding="0" cellspacing="0" width="100%">
                        <tr>
                            <td style="color: white; font-size: 14px;" align="center">
                                &copy; 2024 tabib. All rights reserved.
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """




    # Setting up the MIME
    message = MIMEMultipart("alternative")
    message["From"] = from_email
    message["To"] = to_email
    message["Subject"] = subject

    # Attach the HTML content
    message.attach(MIMEText(html_content, "html"))

    # Creating SMTP session
    try:
        # Use Gmail with TLS
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()  # Enable security
        server.login(from_email, app_password)  # Login with email and app password
        text = message.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        print(f"Email sent successfully to {to_email}")
        return {"Message": f"Email sent successfully to {to_email}", "error": False}
    except Exception as e:
        print(f"Failed to send email. Error: {e}")
        return {"Message": f"Failed to send email. Error {e}", "error": True}
