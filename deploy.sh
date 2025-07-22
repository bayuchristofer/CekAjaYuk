#!/bin/bash

# CekAjaYuk Deployment Script untuk Hostinger VPS
# Jalankan script ini di VPS setelah upload file

echo "🚀 Starting CekAjaYuk deployment on Hostinger VPS..."

# Update sistem
echo "📦 Updating system packages..."
apt update && apt upgrade -y

# Install Python 3.11 dan dependencies
echo "🐍 Installing Python 3.11..."
apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Install sistem dependencies
echo "📚 Installing system dependencies..."
apt install -y nginx supervisor git curl wget unzip
apt install -y tesseract-ocr tesseract-ocr-ind libtesseract-dev
apt install -y build-essential libssl-dev libffi-dev
apt install -y libhdf5-dev pkg-config

# Buat user untuk aplikasi
echo "👤 Creating application user..."
useradd -m -s /bin/bash cekajayuk
usermod -aG www-data cekajayuk

# Buat direktori aplikasi
echo "📁 Creating application directories..."
mkdir -p /var/www/cekajayuk
mkdir -p /var/log/cekajayuk
mkdir -p /var/run/cekajayuk

# Set permissions
chown -R cekajayuk:www-data /var/www/cekajayuk
chown -R cekajayuk:www-data /var/log/cekajayuk
chmod -R 755 /var/www/cekajayuk

# Copy aplikasi ke direktori target
echo "📋 Copying application files..."
cp -r . /var/www/cekajayuk/
cd /var/www/cekajayuk

# Buat virtual environment
echo "🔧 Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Set environment variables
echo "⚙️ Setting up environment..."
cat > .env << EOF
FLASK_ENV=production
PORT=5001
HOST=0.0.0.0
PYTHONPATH=/var/www/cekajayuk
EOF

# Setup Nginx
echo "🌐 Configuring Nginx..."
cp nginx.conf /etc/nginx/sites-available/cekajayuk
ln -sf /etc/nginx/sites-available/cekajayuk /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
nginx -t

# Setup Supervisor untuk Gunicorn
echo "🔄 Setting up Supervisor..."
cat > /etc/supervisor/conf.d/cekajayuk.conf << EOF
[program:cekajayuk]
command=/var/www/cekajayuk/venv/bin/gunicorn -c gunicorn.conf.py backend_working:app
directory=/var/www/cekajayuk
user=cekajayuk
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/cekajayuk/gunicorn.log
environment=PATH="/var/www/cekajayuk/venv/bin"
EOF

# Restart services
echo "🔄 Starting services..."
systemctl restart supervisor
systemctl enable supervisor
systemctl restart nginx
systemctl enable nginx

# Update supervisor
supervisorctl reread
supervisorctl update
supervisorctl start cekajayuk

echo "✅ Deployment completed!"
echo "🌐 Your website should be accessible at: http://YOUR-VPS-IP"
echo "📊 Check status: supervisorctl status cekajayuk"
echo "📝 Check logs: tail -f /var/log/cekajayuk/gunicorn.log"
