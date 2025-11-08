// frontend/package.json
{
  "name": "economic-stress-index-frontend",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "type-check": "tsc --noEmit",
    "test": "jest",
    "test:watch": "jest --watch",
    "e2e": "playwright test"
  },
  "dependencies": {
    "next": "14.0.4",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "@tanstack/react-query": "^5.8.4",
    "framer-motion": "^10.16.16",
    "recharts": "^2.8.0",
    "chart.js": "^4.4.0",
    "react-chartjs-2": "^5.2.0",
    "socket.io-client": "^4.7.4",
    "date-fns": "^2.30.0",
    "lucide-react": "^0.294.0",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.0.0",
    "tailwind-merge": "^2.1.0",
    "@radix-ui/react-alert-dialog": "^1.0.5",
    "@radix-ui/react-toast": "^1.1.5",
    "zustand": "^4.4.7"
  },
  "devDependencies": {
    "typescript": "^5.3.2",
    "@types/node": "^20.9.2",
    "@types/react": "^18.2.38",
    "@types/react-dom": "^18.2.17",
    "tailwindcss": "^3.3.6",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.32",
    "@typescript-eslint/eslint-plugin": "^6.12.0",
    "@typescript-eslint/parser": "^6.12.0",
    "eslint": "^8.54.0",
    "eslint-config-next": "14.0.4",
    "jest": "^29.7.0",
    "@testing-library/react": "^14.1.2",
    "@testing-library/jest-dom": "^6.1.5",
    "playwright": "^1.40.1"
  }
}

// frontend/src/types/index.ts
export interface EconomicIndicator {
  name: string;
  symbol: string;
  value: number;
  change: number;
  normalized_value: number;
  weight: number;
  description: string;
  last_updated: string;
}

export interface StressIndexData {
  stress_index: number;
  base_stress: number;
  ml_stress: number;
  level: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';
  components: Record<string, ComponentData>;
  indicators: Record<string, number>;
  anomaly_score: number;
  timestamp: string;
}

export interface ComponentData {
  value: number;
  normalized: number;
  contribution: number;
  weight: number;
}

export interface HistoricalStressPoint {
  timestamp: string;
  stress_index: number;
  level: string;
  indicators?: Record<string, number>;
}

export interface AlertConfig {
  id: string;
  user_id: string;
  threshold: number;
  indicator?: string;
  email_enabled: boolean;
  sms_enabled: boolean;
  webhook_url?: string;
}

// frontend/src/lib/api.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

class APIClient {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${url}`, error);
      throw error;
    }
  }

  // Stress Index endpoints
  async getCurrentStress(): Promise<StressIndexData> {
    return this.request<StressIndexData>('/stress-index/current');
  }

  async getHistoricalStress(days: number = 30, resolution: string = '1h'): Promise<HistoricalStressPoint[]> {
    return this.request<HistoricalStressPoint[]>(`/stress-index/historical?days=${days}&resolution=${resolution}`);
  }

  // Indicators endpoints
  async getIndicators(): Promise<EconomicIndicator[]> {
    return this.request<EconomicIndicator[]>('/indicators');
  }

  // Alerts endpoints
  async subscribeToAlerts(email: string, threshold: number): Promise<{ message: string }> {
    return this.request('/stress-index/alerts/subscribe', {
      method: 'POST',
      body: JSON.stringify({ email, threshold }),
    });
  }
}

export const apiClient = new APIClient();

// frontend/src/hooks/useWebSocket.ts
import { useEffect, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';
import { StressIndexData } from '@/types';

export function useWebSocket(url: string) {
  const [data, setData] = useState<StressIndexData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    // Create WebSocket connection
    const socket = io(url, {
      transports: ['websocket'],
      upgrade: true,
    });

    socketRef.current = socket;

    socket.on('connect', () => {