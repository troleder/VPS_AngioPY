import os

path = "/etc/nginx/sites-available/angiopysegmentation.tech"
if not os.path.exists(path):
    print("Nginx config not found at " + path)
    exit(1)

with open(path, "r") as f:
    content = f.read()

if "location /admin" in content:
    print("Nginx is already patched. Skipping.")
    exit(0)

patch = """    location /admin {
        alias /var/www/analiza-dicom/admin-dist/;
        index index.html;
        try_files $uri $uri/ /admin/index.html;
    }

    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        client_max_body_size 150M;
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
    }

    location / {"""

new_content = content.replace("    location / {", patch)

with open(path, "w") as f:
    f.write(new_content)

print("Nginx config patched successfully.")
