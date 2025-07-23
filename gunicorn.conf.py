# Gunicorn configuration for CekAjaYuk Production
import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '5001')}"
backlog = 2048

# Worker processes (VPS optimized)
workers = 2  # Reduced for VPS memory constraints
worker_class = "sync"
worker_connections = 500  # Reduced for VPS
timeout = 120
keepalive = 2

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 500  # Reduced for VPS
max_requests_jitter = 50

# Logging
accesslog = "/var/log/cekajayuk/access.log"
errorlog = "/var/log/cekajayuk/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "cekajayuk"

# Server mechanics
daemon = False
pidfile = "/var/run/cekajayuk.pid"
user = "www-data"
group = "www-data"
tmp_upload_dir = None

# SSL (jika menggunakan HTTPS)
# keyfile = "/path/to/ssl/private.key"
# certfile = "/path/to/ssl/certificate.crt"

# Preload application for better performance
preload_app = True

# Memory management
worker_tmp_dir = "/dev/shm"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
