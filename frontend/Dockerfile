#resume-builder/frontend/Dockerfile
# Use a lightweight Nginx image as a base
FROM nginx:alpine

# Copy your static HTML, CSS, and JS files into the web server's root directory
COPY . /usr/share/nginx/html

# The container will listen on port 80 by default, which is what Nginx does.
EXPOSE 80