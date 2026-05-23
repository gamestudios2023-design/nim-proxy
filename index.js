const express = require('express');
const fetch = require('node-fetch');

const app = express();
app.use(express.json({ limit: '10mb' }));

const NIM_BASE = 'https://integrate.api.nvidia.com/v1';

const MODEL_ROUTES = {
  'kimi':           'moonshotai/kimi-k2.6',
  'deepseek':       'deepseek-ai/deepseek-v4-pro',
  'deepseek-flash': 'deepseek-ai/deepseek-v4-flash',
  'glm':            'z-ai/glm-5.1',
  'qwen':           'qwen/qwen3.5-397b-a17b',
  'qwen-coder':     'qwen/qwen3-coder-480b-a35b-instruct',
  'llama':          'meta/llama-3.3-70b-instruct',
  'llama4':         'meta/llama-4-maverick-17b-128e-instruct',
  'nemotron':       'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'mistral':        'mistralai/mistral-large-3-675b-instruct-2512',
  'minimax':        'minimaxai/minimax-m2.7',
  'gemma':          'google/gemma-4-31b-it',
};

// Fake Anthropic models list — Claude Code validates against this
const ANTHROPIC_MODELS = [
  'claude-opus-4-7',
  'claude-sonnet-4-6',
  'claude-haiku-4-5-20251001',
  'claude-opus-4-5',
  'claude-sonnet-4-5',
  'claude-3-5-sonnet-20241022',
  'claude-3-5-haiku-20241022',
  'claude-3-opus-20240229',
].map(id => ({ type: 'model', id, display_name: id, created_at: '2024-01-01' }));

app.get('/', (req, res) => {
  res.json({
    status: 'ok',
    service: 'nim-proxy',
    routes: Object.entries(MODEL_ROUTES).map(([a, m]) => ({ alias: `/${a}/v1`, model: m })),
  });
});

// Return fake Anthropic models so Claude Code doesn't complain
app.get(['/:alias/v1/models', '/v1/models'], (req, res) => {
  res.json({ data: ANTHROPIC_MODELS });
});

// --- Format converters ---

function anthropicToOpenAI(body, modelOverride) {
  const messages = [];
  if (body.system) {
    const sys = typeof body.system === 'string'
      ? body.system
      : body.system.map(b => b.text || '').join('');
    messages.push({ role: 'system', content: sys });
  }
  for (const msg of (body.messages || [])) {
    const content = typeof msg.content === 'string'
      ? msg.content
      : (msg.content || []).filter(b => b.type === 'text').map(b => b.text).join('');
    messages.push({ role: msg.role, content });
  }
  const oai = {
    model: modelOverride,
    messages,
    max_tokens: body.max_tokens || 1024,
    stream: body.stream || false,
  };
  if (body.temperature != null) oai.temperature = body.temperature;
  if (body.top_p != null) oai.top_p = body.top_p;
  return oai;
}

function openAIToAnthropic(oaiRes, model) {
  const choice = oaiRes.choices?.[0] || {};
  return {
    id: oaiRes.id || `msg_${Date.now()}`,
    type: 'message',
    role: 'assistant',
    content: [{ type: 'text', text: choice.message?.content || '' }],
    model,
    stop_reason: choice.finish_reason === 'stop' ? 'end_turn' : (choice.finish_reason || 'end_turn'),
    stop_sequence: null,
    usage: {
      input_tokens: oaiRes.usage?.prompt_tokens || 0,
      output_tokens: oaiRes.usage?.completion_tokens || 0,
    },
  };
}

// Stream: pipe OpenAI SSE → Anthropic SSE
async function streamOpenAIToAnthropic(nimRes, res, model, msgId) {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('X-Accel-Buffering', 'no');

  let started = false;
  let outputTokens = 0;
  let buf = '';

  const write = (event, data) => res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

  nimRes.body.on('data', chunk => {
    buf += chunk.toString();
    const lines = buf.split('\n');
    buf = lines.pop();

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const raw = line.slice(6).trim();
      if (raw === '[DONE]') {
        res.write(`event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n`);
        write('message_delta', {
          type: 'message_delta',
          delta: { stop_reason: 'end_turn', stop_sequence: null },
          usage: { output_tokens: outputTokens },
        });
        res.write(`event: message_stop\ndata: {"type":"message_stop"}\n\n`);
        return;
      }
      let chunk_data;
      try { chunk_data = JSON.parse(raw); } catch { continue; }

      if (!started) {
        started = true;
        write('message_start', {
          type: 'message_start',
          message: {
            id: chunk_data.id || msgId,
            type: 'message', role: 'assistant', content: [], model,
            stop_reason: null, stop_sequence: null,
            usage: { input_tokens: 0, output_tokens: 0 },
          },
        });
        write('content_block_start', {
          type: 'content_block_start', index: 0,
          content_block: { type: 'text', text: '' },
        });
        res.write(`event: ping\ndata: {"type":"ping"}\n\n`);
      }

      const text = chunk_data.choices?.[0]?.delta?.content;
      if (text) {
        outputTokens++;
        write('content_block_delta', {
          type: 'content_block_delta', index: 0,
          delta: { type: 'text_delta', text },
        });
      }
    }
  });

  nimRes.body.on('end', () => res.end());
  nimRes.body.on('error', err => {
    console.error('Stream error:', err.message);
    res.end();
  });
}

// --- Main handler ---

app.all(['/:alias/v1/*', '/v1/*'], async (req, res) => {
  const secret = process.env.PROXY_SECRET;
  const isClaudeCode = req.headers['anthropic-version'] != null;
  if (secret && !isClaudeCode && req.headers['x-proxy-secret'] !== secret) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const alias = req.params.alias;
  const modelOverride = alias ? MODEL_ROUTES[alias] : null;

  const fullPath = req.path;
  const v1Index = fullPath.indexOf('/v1');
  const nimPath = fullPath.slice(v1Index + 3); // e.g. /messages or /chat/completions

  const isAnthropicMessages = nimPath === '/messages';

  let nimUrl, nimBody;

  if (isAnthropicMessages) {
    // Claude Code → Anthropic format → translate to OpenAI
    nimUrl = `${NIM_BASE}/chat/completions`;
    nimBody = anthropicToOpenAI(req.body, modelOverride || 'meta/llama-3.3-70b-instruct');
  } else {
    // Direct OpenAI-style call (tests, etc.)
    nimUrl = `${NIM_BASE}${nimPath}`;
    nimBody = { ...req.body };
    if (modelOverride) nimBody.model = modelOverride;
  }

  try {
    const nimRes = await fetch(nimUrl, {
      method: req.method,
      headers: {
        'Authorization': `Bearer ${process.env.NIM_API_KEY}`,
        'Content-Type': 'application/json',
        'Accept': req.headers['accept'] || 'application/json',
      },
      body: ['POST', 'PUT', 'PATCH'].includes(req.method) ? JSON.stringify(nimBody) : undefined,
    });

    if (isAnthropicMessages && nimBody.stream) {
      await streamOpenAIToAnthropic(nimRes, res, modelOverride || nimBody.model, `msg_${Date.now()}`);
    } else if (isAnthropicMessages) {
      const oaiData = await nimRes.json();
      res.status(nimRes.status).json(openAIToAnthropic(oaiData, modelOverride || nimBody.model));
    } else {
      // Pass-through for /chat/completions, /models, etc.
      res.status(nimRes.status);
      nimRes.headers.forEach((v, k) => {
        if (!['content-encoding', 'transfer-encoding', 'connection'].includes(k)) res.setHeader(k, v);
      });
      res.setHeader('X-Accel-Buffering', 'no');
      nimRes.body.pipe(res);
    }
  } catch (err) {
    console.error('Proxy error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

app.listen(process.env.PORT || 3000, () =>
  console.log('NIM proxy on port', process.env.PORT || 3000)
);
