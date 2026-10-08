import os, smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()
host = os.getenv("SMTP_HOST", "smtp.gmail.com")
port = int(os.getenv("SMTP_PORT", "587"))
user = os.getenv("SMTP_USER")
pw = os.getenv("SMTP_PASSWORD")
print("host", host, "port", port, "user", repr(user), "password length", len(pw or ""))

msg = EmailMessage()
msg["From"], msg["To"], msg["Subject"] = user, user, "smtp test"
msg.set_content("ok")
try:
    if port == 465:
        s = smtplib.SMTP_SSL(host, port, timeout=15)
    else:
        s = smtplib.SMTP(host, port, timeout=15)
        s.starttls()
    s.login(user, pw)
    s.send_message(msg)
    s.quit()
    print("SENT OK. Check the inbox of", user)
except Exception as e:
    print("FAILED:", repr(e))