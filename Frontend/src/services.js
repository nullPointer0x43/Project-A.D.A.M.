import { v4 as uuidv4 } from 'uuid';

const BASE_URL = "/api";

const isSecure = window.location.protocol === 'https:';
const wsProtocol = isSecure ? 'wss:' : 'ws:';

const WS_URL = `${wsProtocol}//${window.location.host}/ws/pipeline`;

export const sessionService = {
  getThreadId: () => {
    let threadId = localStorage.getItem('agent_thread_id');
    if (!threadId) {
      threadId = `thread_${uuidv4()}`;
      localStorage.setItem('agent_thread_id', threadId);
    }
    return threadId;
  },

  clearSession: () => {
    localStorage.removeItem('agent_thread_id');
  },

  async resetPipeline() {
    const threadId = this.getThreadId();
    try {
      const response = await fetch(`${BASE_URL}/reset/${threadId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || "Failed to trigger back-end pipeline reset.");
      }

      this.clearSession();
      
      return response.json();
    } catch (err) {
      console.error("Pipeline Reset Service Error:", err);
      throw err;
    }
  }
};

export const apiService = {
  async downloadReport() {
    try {
      const threadId = sessionService.getThreadId();
      const response = await fetch(`${BASE_URL}/generate-report?thread_id=${threadId}`);
      if (!response.ok) throw new Error("Failed to download PDF");

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');

      link.href = url;
      link.setAttribute('download', `Report_${threadId}.pdf`);
      document.body.appendChild(link);
      link.click();
      
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Download service error:", err);
      throw err;
    }
  },

  async getPageData(pageNumber) {
    const threadId = sessionService.getThreadId();
    const response = await fetch(`${BASE_URL}/page-data/?page=${pageNumber}&thread_id=${threadId}`);
    if (!response.ok) throw new Error("Failed to fetch page data");
    return response.json();
  },

  createWebSocket: () => {
    const threadId = sessionService.getThreadId();
    return new WebSocket(`${WS_URL}/${threadId}`);
  }
};

export const fileService = {
  async uploadDataset(file) {
    const threadId = sessionService.getThreadId();
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${BASE_URL}/upload/${threadId}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const { detail } = await response.json();
        throw new Error(detail || "Backend failed to accept file.");
      }

      return response.json();
    } catch (err) {
      console.error("Upload Service Error:", err);
      throw err;
    }
  },

  async resetDataset() {
    const threadId = sessionService.getThreadId();
    try {
      const response = await fetch(`${BASE_URL}/upload/${threadId}`, { method: 'DELETE' });
      if (!response.ok) throw new Error("Failed to reset dataset state.");
      return response.json();
    } catch (err) {
      console.error("Reset Service Error:", err);
      throw err;
    }
  }
};

export const multivariateService = {
  parsePairId: (pairId) => pairId.replace(/[()']/g, '').split(', '),

  async fetchPlotData(var1, var2) {
    const threadId = sessionService.getThreadId();
    try {
      const response = await fetch(`${BASE_URL}/plot-data/?var1=${var1}&var2=${var2}&thread_id=${threadId}`);
      return response.ok ? response.json() : [];
    } catch (err) {
      console.error("Fetch error:", err);
      return [];
    }
  }
};

export const insightService = {
  async getInsightDetails(name) {
    const threadId = sessionService.getThreadId();
    try {
      const response = await fetch(`${BASE_URL}/insite-data/?name=${encodeURIComponent(name)}&thread_id=${threadId}`);
      if (!response.ok) throw new Error(`Insight API Error: ${response.statusText}`);
      return response.json();
    } catch (error) {
      console.error("Service Error while fetching insight:", error);
      throw error;
    }
  }
};