const express = require('express');
const fetch = require('node-fetch');

const app = express();
app.use(express.json({ limit: '10mb' }));

const NIM_BASE = 'https://integrate.api.nvidia.com/v1';

// Model routing table — /<alias>/v1/* → NIM model ID
const MODEL_ROUTES = {
  'kimi':          'moonshotai/kimi-k2.6',
  'deepseek':      'deepseek-ai/deepseek-v4-pro',
  'deepseek-flash':'deepseek-ai/deepseek-v4-flash',
  'glm':           'z-ai/glm-5.1',
  'qwen':          'qwen/qwen3.5-397b-a17b',
  'qwen-coder':    'qwen/qwen3-coder-480b-a35b-instruct',
  'llama':         'meta/llama-3.3-70b-instruct',
  'llama4':        'meta/llama-4-maverick-17b-128e-instruct',
  'nemotron':      'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'mistral':       'mistralai/mistral-large-3-675b-instruct-2512',
  'minimax':       'minimaxai/minimax-m2.7',
  'gemma':         'google/gemma-4-31b-it',
};

app.get('/', (req, res) => {
  const routes = Object.entries(MODEL_ROUTES).map(([alias, model]) => ({
    alias: `/${alias}/v1`,
    model,
  }));
  res.json({ status: 'ok', service: 'nim-proxy', routes });
});

// Handle both /<alias>/v1/* and /v1/* (default: llama-3.3-70b)
app.all(['/:alias/v1/*', '/v1/*'], async (req, res) => {
  const secret = process.env.PROXY_SECRET;
  if (secret && req.headers['x-proxy-secret'] !== secret) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  // Determine model from route alias or keep body model
  const alias = req.params.alias;
  const forcedModel = alias ? MODEL_ROUTES[alias] : null;

  // Extract the /v1/* sub-path
  const fullPath = req.path; // e.g. /kimi/v1/chat/completions or /v1/chat/completions
  const v1Index = fullPath.indexOf('/v1');
  const nimPath = fullPath.slice(v1Index + 3); // → /chat/completions

  const url = `${NIM_BASE}${nimPath}`;

  // Override model in body if alias-based routing
  const body = { ...req.body };
  if (forcedModel) body.model = forcedModel;

  try {
    const nimRes = await fetch(url, {
      method: req.method,
      headers: {
        'Authorization': `Bearer ${process.env.NIM_API_KEY}`,
        'Content-Type': 'application/json',
        'Accept': req.headers['accept'] || 'application/json',
      },
      body: ['POST', 'PUT', 'PATCH'].includes(req.method)
        ? JSON.stringify(body)
        : undefined,
    });

    res.status(nimRes.status);
    nimRes.headers.forEach((v, k) => {
      if (!['content-encoding', 'transfer-encoding', 'connection'].includes(k)) {
        res.setHeader(k, v);
      }
    });
    res.setHeader('X-Accel-Buffering', 'no');
    nimRes.body.pipe(res);
  } catch (err) {
    console.error('Proxy error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

app.listen(process.env.PORT || 3000, () =>
  console.log('NIM proxy on port', process.env.PORT || 3000)
);
