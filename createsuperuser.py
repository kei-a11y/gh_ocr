# create_superuser.py
import os
import sys
from django.contrib.auth import get_user_model

username = os.environ.get("DJANGO_SUPERUSER_USERNAME")
email = os.environ.get("DJANGO_SUPERUSER_EMAIL", "")
password = os.environ.get("DJANGO_SUPERUSER_PASSWORD")

if not username or not password:
    print("Missing DJANGO_SUPERUSER_USERNAME or DJANGO_SUPERUSER_PASSWORD. Skipping.")
    sys.exit(0)

User = get_user_model()
if not User.objects.filter(username=username).exists():
    User.objects.create_superuser(username=username, email=email, password=password)
    print(f"Created superuser: {username}")
else:
    print(f"Superuser '{username}' already exists. Skipping.")
