# Frontend Dockerfile for arXiv Paper Curator
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files for dependency installation
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy all source code
COPY . .

# Expose React dev server port
EXPOSE 3000

# Set environment variable for React dev server
ENV CHOKIDAR_USEPOLLING=true
ENV WDS_SOCKET_PORT=0

# Start React development server
CMD ["npm", "start"]

