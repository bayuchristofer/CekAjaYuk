# 🚀 Panduan Deployment CekAjaYuk ke Hostinger VPS

## 📋 Persiapan

### 1. Akses VPS Anda
```bash
ssh root@IP-VPS-ANDA
```

### 2. Upload File Aplikasi
Gunakan SCP atau FileZilla untuk upload semua file ke VPS:
```bash
scp -r . root@IP-VPS-ANDA:/root/cekajayuk/
```

## 🔧 Instalasi Otomatis

### 1. Jalankan Script Deployment
```bash
cd /root/cekajayuk
chmod +x deploy.sh
./deploy.sh
```

Script ini akan:
- ✅ Update sistem
- ✅ Install Python 3.11 dan dependencies
- ✅ Install Nginx dan Supervisor
- ✅ Setup virtual environment
- ✅ Install packages Python
- ✅ Konfigurasi Nginx dan Gunicorn
- ✅ Start semua services

## 🔧 Instalasi Manual (Jika Diperlukan)

### 1. Update Sistem
```bash
apt update && apt upgrade -y
```

### 2. Install Dependencies
```bash
# Python dan tools
apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Web server dan process manager
apt install -y nginx supervisor

# OCR dan ML dependencies
apt install -y tesseract-ocr tesseract-ocr-ind libtesseract-dev
apt install -y build-essential libssl-dev libffi-dev libhdf5-dev
```

### 3. Setup Aplikasi
```bash
# Buat direktori
mkdir -p /var/www/cekajayuk
cp -r . /var/www/cekajayuk/
cd /var/www/cekajayuk

# Virtual environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Konfigurasi Nginx
```bash
cp nginx.conf /etc/nginx/sites-available/cekajayuk
ln -s /etc/nginx/sites-available/cekajayuk /etc/nginx/sites-enabled/
rm /etc/nginx/sites-enabled/default
nginx -t
systemctl restart nginx
```

### 5. Setup Supervisor
```bash
cp supervisor.conf /etc/supervisor/conf.d/cekajayuk.conf
supervisorctl reread
supervisorctl update
supervisorctl start cekajayuk
```

## 🌐 Akses Website

Setelah deployment selesai:
- **HTTP**: `http://IP-VPS-ANDA`
- **Health Check**: `http://IP-VPS-ANDA/api/health`

## 📊 Monitoring

### Cek Status Services
```bash
# Status aplikasi
supervisorctl status cekajayuk

# Status Nginx
systemctl status nginx

# Cek logs
tail -f /var/log/cekajayuk/gunicorn.log
tail -f /var/log/nginx/error.log
```

### Restart Services
```bash
# Restart aplikasi
supervisorctl restart cekajayuk

# Restart Nginx
systemctl restart nginx
```

## 🔒 Setup SSL (Opsional)

### 1. Install Certbot
```bash
apt install -y certbot python3-certbot-nginx
```

### 2. Dapatkan SSL Certificate
```bash
certbot --nginx -d your-domain.com -d www.your-domain.com
```

### 3. Auto-renewal
```bash
crontab -e
# Tambahkan line ini:
0 12 * * * /usr/bin/certbot renew --quiet
```

## 🛠️ Troubleshooting

### Aplikasi tidak bisa diakses
```bash
# Cek status
supervisorctl status cekajayuk
systemctl status nginx

# Cek logs
tail -f /var/log/cekajayuk/gunicorn.log
tail -f /var/log/nginx/error.log
```

### Error loading models
```bash
# Pastikan file model ada
ls -la /var/www/cekajayuk/models/

# Cek permissions
chown -R cekajayuk:www-data /var/www/cekajayuk
```

### Memory issues
```bash
# Cek memory usage
free -h
htop

# Kurangi workers di gunicorn.conf.py jika perlu
```

## 📝 Maintenance

### Update Aplikasi
```bash
cd /var/www/cekajayuk
git pull  # jika menggunakan git
supervisorctl restart cekajayuk
```

### Backup
```bash
# Backup aplikasi
tar -czf cekajayuk-backup-$(date +%Y%m%d).tar.gz /var/www/cekajayuk

# Backup database (jika ada)
# mysqldump atau pg_dump sesuai kebutuhan
```

## 🎯 Optimasi Performance

1. **Gunakan CDN** untuk static files
2. **Setup Redis** untuk caching (opsional)
3. **Monitor resource usage** dengan htop/netdata
4. **Optimize model loading** dengan lazy loading

---

**🎉 Selamat! Website CekAjaYuk Anda sudah live di VPS Hostinger!**
