#!/bin/bash

# AI DevOps Platform Docker Runner
# This script helps run the application with Docker

set -e

echo "🚀 AI DevOps Platform Docker Setup"
echo "=================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first:"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "📋 Setting up environment variables..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Copied .env.example to .env"
        echo "⚠️  Please edit .env with your actual credentials"
    else
        echo "❌ .env.example not found"
        exit 1
    fi
fi

echo "🏗️  Building and starting containers..."
echo "📝 Use Ctrl+C to stop"

# Try Docker Compose V2 first, fallback to V1
if command -v "docker compose" &> /dev/null; then
    docker compose up --build
elif command -v docker-compose &> /dev/null; then
    docker-compose up --build
else
    echo "❌ Docker Compose not found"
    exit 1
fi