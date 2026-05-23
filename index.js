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
    if (typeof msg.content === 'string') {
      messages.push({ role: msg.role, content: msg.content });
      continue;
    }
    const blocks = msg.content || [];
    const textParts = [];
    const toolCalls = [];
    for (const b of blocks) {
      if (b.type === 'text') textParts.push(b.text);
      else if (b.type === 'tool_use') {
        toolCalls.push({
          id: b.id,
          type: 'function',
          function: { name: b.name, arguments: JSON.stringify(b.input || {}) },
        });
      } else if (b.type === 'tool_result') {
        // Emit a separate tool role message
        const resultContent = typeof b.content === 'string'
          ? b.content
          : (b.content || []).map(c => c.text || JSON.stringify(c)).join('\n');
        messages.push({
          role: 'tool',
          tool_call_id: b.tool_use_id,
          content: resultContent,
        });
      }
    }
    if (textParts.length || toolCalls.length) {
      const m = { role: msg.role, content: textParts.join('') || null };
      if (toolCalls.length) m.tool_calls = toolCalls;
      messages.push(m);
    }
  }
  const oai = {
    model: modelOverride,
    messages,
    max_tokens: body.max_tokens || 1024,
    stream: body.stream || false,
  };
  if (body.temperature != null) oai.temperature = body.temperature;
  if (body.top_p != null) oai.top_p = body.top_p;
  if (body.tools && body.tools.length) {
    oai.tools = body.tools.map(t => ({
      type: 'function',
      function: {
        name: t.name,
        description: t.description,
        parameters: t.input_schema,
      },
    }));
    if (body.tool_choice) {
      if (body.tool_choice.type === 'auto') oai.tool_choice = 'auto';
      else if (body.tool_choice.type === 'any') oai.tool_choice = 'required';
      else if (body.tool_choice.type === 'tool') {
        oai.tool_choice = { type: 'function', function: { name: body.tool_choice.name } };
      }
    }
  }
  return oai;
}

function openAIToAnthropic(oaiRes, model) {
  const choice = oaiRes.choices?.[0] || {};
  const msg = choice.message || {};
  const content = [];
  if (msg.content) content.push({ type: 'text', text: msg.content });
  if (msg.tool_calls) {
    for (const tc of msg.tool_calls) {
      let input = {};
      try { input = JSON.parse(tc.function.arguments || '{}'); } catch {}
      content.push({
        type: 'tool_use',
        id: tc.id,
        name: tc.function.name,
        input,
      });
    }
  }
  if (!content.length) content.push({ type: 'text', text: '' });
  const stopReason = choice.finish_reason === 'tool_calls' ? 'tool_use'
    : choice.finish_reason === 'stop' ? 'end_turn'
    : choice.finish_reason === 'length' ? 'max_tokens'
    : (choice.finish_reason || 'end_turn');
  return {
    id: oaiRes.id || `msg_${Date.now()}`,
    type: 'message',
    role: 'assistant',
    content,
    model,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: oaiRes.usage?.prompt_tokens || 0,
      output_tokens: oaiRes.usage?.completion_tokens || 0,
    },
  };
}

// Stream: pipe OpenAI SSE → Anthropic SSE (handles text + tool_calls)
async function streamOpenAIToAnthropic(nimRes, res, model, msgId) {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('X-Accel-Buffering', 'no');

  let messageStarted = false;
  let textBlockStarted = false;
  let textBlockClosed = false;
  let outputTokens = 0;
  let finalStopReason = 'end_turn';
  let buf = '';
  const toolBlocks = {}; // tcIndex → { anthropicIndex, started, closed, id, name }
  let nextAnthropicIndex = 0;

  const write = (event, data) => res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

  const startMessage = (id) => {
    if (messageStarted) return;
    messageStarted = true;
    write('message_start', {
      type: 'message_start',
      message: {
        id: id || msgId,
        type: 'message', role: 'assistant', content: [], model,
        stop_reason: null, stop_sequence: null,
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    });
    res.write(`event: ping\ndata: {"type":"ping"}\n\n`);
  };

  const startTextBlock = () => {
    if (textBlockStarted) return;
    textBlockStarted = true;
    write('content_block_start', {
      type: 'content_block_start', index: 0,
      content_block: { type: 'text', text: '' },
    });
    nextAnthropicIndex = Math.max(nextAnthropicIndex, 1);
  };

  const closeTextBlock = () => {
    if (textBlockStarted && !textBlockClosed) {
      textBlockClosed = true;
      write('content_block_stop', { type: 'content_block_stop', index: 0 });
    }
  };

  nimRes.body.on('data', chunk => {
    buf += chunk.toString();
    const lines = buf.split('\n');
    buf = lines.pop();

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const raw = line.slice(6).trim();
      if (raw === '[DONE]') {
        closeTextBlock();
        for (const tk of Object.keys(toolBlocks)) {
          const tb = toolBlocks[tk];
          if (tb.started && !tb.closed) {
            tb.closed = true;
            write('content_block_stop', { type: 'content_block_stop', index: tb.anthropicIndex });
          }
        }
        write('message_delta', {
          type: 'message_delta',
          delta: { stop_reason: finalStopReason, stop_sequence: null },
          usage: { output_tokens: outputTokens },
        });
        res.write(`event: message_stop\ndata: {"type":"message_stop"}\n\n`);
        return;
      }
      let cd;
      try { cd = JSON.parse(raw); } catch { continue; }
      startMessage(cd.id);

      const choice = cd.choices?.[0] || {};
      const delta = choice.delta || {};

      // Text content
      if (delta.content) {
        startTextBlock();
        outputTokens++;
        write('content_block_delta', {
          type: 'content_block_delta', index: 0,
          delta: { type: 'text_delta', text: delta.content },
        });
      }

      // Tool calls
      if (delta.tool_calls) {
        for (const tc of delta.tool_calls) {
          const tcIdx = tc.index ?? 0;
          let tb = toolBlocks[tcIdx];
          if (!tb) {
            // Need to close text block before starting tool block
            closeTextBlock();
            tb = toolBlocks[tcIdx] = {
              anthropicIndex: nextAnthropicIndex++,
              started: false, closed: false,
              id: tc.id || `toolu_${Date.now()}_${tcIdx}`,
              name: tc.function?.name || '',
            };
          }
          if (tc.function?.name) tb.name = tc.function.name;
          if (tc.id) tb.id = tc.id;

          if (!tb.started && tb.name) {
            tb.started = true;
            write('content_block_start', {
              type: 'content_block_start',
              index: tb.anthropicIndex,
              content_block: { type: 'tool_use', id: tb.id, name: tb.name, input: {} },
            });
          }

          const args = tc.function?.arguments;
          if (args && tb.started) {
            write('content_block_delta', {
              type: 'content_block_delta',
              index: tb.anthropicIndex,
              delta: { type: 'input_json_delta', partial_json: args },
            });
          }
        }
      }

      // Track finish reason
      if (choice.finish_reason) {
        finalStopReason = choice.finish_reason === 'tool_calls' ? 'tool_use'
          : choice.finish_reason === 'stop' ? 'end_turn'
          : choice.finish_reason === 'length' ? 'max_tokens'
          : 'end_turn';
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
  const alias = req.params.alias || 'default';
  const model = alias !== 'default' ? (MODEL_ROUTES[alias] || alias) : 'meta/llama-3.3-70b-instruct';
  const fullPath = req.path;
  const v1Index = fullPath.indexOf('/v1');
  const nimPath = fullPath.slice(v1Index + 3);
  console.log(`[${new Date().toISOString()}] ${req.method} /${alias}/v1${nimPath} → ${model}`);

  const secret = process.env.PROXY_SECRET;
  const isClaudeCode = req.headers['anthropic-version'] != null;
  if (secret && !isClaudeCode && req.headers['x-proxy-secret'] !== secret) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const isAnthropicMessages = nimPath === '/messages';

  let nimUrl, nimBody;

  if (isAnthropicMessages) {
    nimUrl = `${NIM_BASE}/chat/completions`;
    nimBody = anthropicToOpenAI(req.body, model);
  } else {
    nimUrl = `${NIM_BASE}${nimPath}`;
    nimBody = { ...req.body };
    if (alias !== 'default') nimBody.model = model;
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);

  try {
    const nimRes = await fetch(nimUrl, {
      method: req.method,
      headers: {
        'Authorization': `Bearer ${process.env.NIM_API_KEY}`,
        'Content-Type': 'application/json',
        'Accept': req.headers['accept'] || 'application/json',
      },
      body: ['POST', 'PUT', 'PATCH'].includes(req.method) ? JSON.stringify(nimBody) : undefined,
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (isAnthropicMessages && nimBody.stream) {
      await streamOpenAIToAnthropic(nimRes, res, model, `msg_${Date.now()}`);
    } else if (isAnthropicMessages) {
      const oaiData = await nimRes.json();
      res.status(nimRes.status).json(openAIToAnthropic(oaiData, model));
    } else {
      res.status(nimRes.status);
      nimRes.headers.forEach((v, k) => {
        if (!['content-encoding', 'transfer-encoding', 'connection'].includes(k)) res.setHeader(k, v);
      });
      res.setHeader('X-Accel-Buffering', 'no');
      nimRes.body.pipe(res);
    }
  } catch (err) {
    clearTimeout(timeout);
    console.error('Proxy error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

app.listen(process.env.PORT || 3000, () =>
  console.log('NIM proxy on port', process.env.PORT || 3000)
);
