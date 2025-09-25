#!/bin/bash

# AI Trading System Deployment Script
# This script deploys the application using Docker Compose

set -e

echo "🚀 Starting AI Trading System deployment..."

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p data

# Build and start services
echo "🏗️  Building and starting services..."
docker-compose down --remove-orphans
docker-compose build --no-cache
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
echo "🔍 Checking service health..."
if docker-compose ps | grep -q "Up"; then
    echo "✅ Services are running successfully!"
    echo ""
    echo "🌐 API available at: http://localhost:8000"
    echo "📊 API documentation at: http://localhost:8000/docs"
    echo "🔴 Redis available at: localhost:6379"
    echo "🐘 PostgreSQL available at: localhost:5432"
    echo ""
    echo "📋 Useful commands:"
    echo "  View logs: docker-compose logs -f"
    echo "  Stop services: docker-compose down"
    echo "  Restart: docker-compose restart"
else
    echo "❌ Some services failed to start. Check logs with: docker-compose logs"
    exit 1
fi

echo "🎉 Deployment completed successfully!"