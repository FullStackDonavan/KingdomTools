# Dockerfile
FROM node:20

WORKDIR /app

COPY package*.json ./
RUN npm install

# Add Prisma CLI globally so `npx` works cleanly
RUN npm install -g prisma

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]
