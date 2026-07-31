import os
from dotenv import load_dotenv

load_dotenv()

IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", 4002))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", 1))