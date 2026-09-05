# Production image: compiles TypeScript in a builder stage, then ships only
# the compiled output and production dependencies in a slim runtime stage.

FROM node:20-alpine AS builder
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY tsconfig.json tsconfig.build.json ./
COPY src ./src
RUN npm run build


FROM node:20-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production

COPY package.json package-lock.json ./
RUN npm ci --omit=dev && npm cache clean --force

COPY --from=builder /app/dist ./dist

RUN addgroup -S app && adduser -S app -G app
USER app

CMD ["node", "dist/index.js"]
