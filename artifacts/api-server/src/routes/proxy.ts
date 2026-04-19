import { Router } from "express";
import { db } from "@workspace/db";
import { apiKeysTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import axios, { type AxiosInstance } from "axios";
import { withSapiom } from "@sapiom/axios";
import { createFetch } from "@sapiom/fetch";

const router = Router();

const SAPIOM_BASE = "https://openrouter.services.sapiom.ai";

// ── Client caches ────────────────────────────────────────────────────────────
// Each API key gets its own persistent client. Mirrors the original server.js
// pattern of creating a single client at startup so transaction state is preserved.

const axiosCache = new Map<string, AxiosInstance>();
function getAxiosClient(apiKey: string): AxiosInstance {
  if (!axiosCache.has(apiKey)) {
    axiosCache.set(apiKey, withSapiom(axios.create(), { apiKey }));
  }
  return axiosCache.get(apiKey)!;
}

type SapiomFetch = ReturnType<typeof createFetch>;
const fetchCache = new Map<string, SapiomFetch>();
function getFetchClient(apiKey: string): SapiomFetch {
  if (!fetchCache.has(apiKey)) {
    fetchCache.set(apiKey, createFetch({ apiKey }));
  }
  return fetchCache.get(apiKey)!;
}

// ── Key selection ─────────────────────────────────────────────────────────────
async function getActiveKey(): Promise<string | null> {
  const keys = await db
    .select({ key: apiKeysTable.key })
    .from(apiKeysTable)
    .where(eq(apiKeysTable.isActive, true));
  if (keys.length === 0) return null;
  return keys[Math.floor(Math.random() * keys.length)].key;
}

// ── GET /v1/models ────────────────────────────────────────────────────────────
router.get("/v1/models", async (req, res) => {
  try {
    const apiKey = await getActiveKey();
    if (!apiKey) {
      res.status(503).json({ error: { message: "No active API keys available" } });
      return;
    }
    const client = getAxiosClient(apiKey);
    const response = await client.get(`${SAPIOM_BASE}/v1/models`);
    res.json(response.data);
  } catch {
    const fallbackModels = [
      "openai/gpt-4o-mini", "openai/gpt-4o", "openai/gpt-4.1-mini", "openai/gpt-4.1-nano",
      "anthropic/claude-sonnet-4", "anthropic/claude-haiku-3.5",
      "google/gemini-2.5-flash", "google/gemini-2.5-pro",
      "meta-llama/llama-4-maverick", "meta-llama/llama-4-scout",
    ].map((id) => ({ id, object: "model" }));
    res.json({ object: "list", data: fallbackModels });
  }
});

// ── POST /v1/chat/completions ─────────────────────────────────────────────────
router.post("/v1/chat/completions", async (req, res) => {
  const body = req.body;
  const isStream = body.stream === true;

  req.log.info({ model: body.model, stream: isStream }, "chat/completions");

  const apiKey = await getActiveKey();
  if (!apiKey) {
    res.status(503).json({ error: { message: "No active API keys available" } });
    return;
  }

  if (isStream) {
    // Use @sapiom/fetch for streaming — it handles x402 via fetch's native response,
    // so the 402 body is always read as JSON (never as a stream), and the retry
    // carries proper payment headers before starting the actual SSE stream.
    const sapiomFetch = getFetchClient(apiKey);

    try {
      const upstream = await sapiomFetch(`${SAPIOM_BASE}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify(body),
      });

      if (!upstream.ok) {
        const errBody = await upstream.json().catch(() => ({
          error: { message: `Upstream error ${upstream.status}` },
        }));
        res.status(upstream.status).json(errBody);
        return;
      }

      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      res.setHeader("X-Accel-Buffering", "no");
      res.flushHeaders();

      // Keep-alive ping every 20 s to beat Replit's 300s proxy timeout
      const keepAlive = setInterval(() => {
        if (!res.writableEnded) res.write(": ping\n\n");
      }, 20000);

      const cleanup = () => {
        clearInterval(keepAlive);
      };

      if (!upstream.body) {
        cleanup();
        res.end();
        return;
      }

      // Pipe the ReadableStream to Express response using Node.js stream API
      const reader = upstream.body.getReader();
      const pump = async () => {
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            if (!res.writableEnded) res.write(value);
          }
        } catch (err) {
          req.log.error({ err }, "stream read error");
        } finally {
          cleanup();
          if (!res.writableEnded) res.end();
        }
      };

      req.on("close", () => {
        cleanup();
        reader.cancel().catch(() => {});
      });

      pump();
    } catch (err: any) {
      req.log.error({ err: err.message }, "stream request failed");
      if (!res.headersSent) {
        res.status(500).json({ error: { message: err.message } });
      } else if (!res.writableEnded) {
        res.end();
      }
    }
  } else {
    // Non-streaming — use @sapiom/axios (works perfectly for JSON responses)
    try {
      const client = getAxiosClient(apiKey);
      const response = await client.post(`${SAPIOM_BASE}/v1/chat/completions`, body);
      res.json(response.data);
    } catch (err: any) {
      const status = err.response?.status || 500;
      const data = err.response?.data || { error: { message: err.message } };
      res.status(status).json(data);
    }
  }
});

// ── POST /v1/embeddings ───────────────────────────────────────────────────────
router.post("/v1/embeddings", async (req, res) => {
  const body = req.body;
  req.log.info({ model: body.model }, "embeddings");

  try {
    const apiKey = await getActiveKey();
    if (!apiKey) {
      res.status(503).json({ error: { message: "No active API keys available" } });
      return;
    }
    const client = getAxiosClient(apiKey);
    const response = await client.post(`${SAPIOM_BASE}/v1/embeddings`, body);
    res.json(response.data);
  } catch (err: any) {
    const status = err.response?.status || 500;
    const data = err.response?.data || { error: { message: err.message } };
    res.status(status).json(data);
  }
});

export default router;
