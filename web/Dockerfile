FROM node:20-alpine

WORKDIR /app

ENV NEXT_TELEMETRY_DISABLED=1

COPY package.json ./
COPY package-lock.json* ./
RUN npm install

COPY . ./

CMD ["npm", "run", "dev", "--", "-H", "0.0.0.0", "-p", "3000"]
