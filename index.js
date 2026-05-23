const express = require('express');
const fetch = require('node-fetch');

const app = express();
app.use(express.json({ limit: '10mb' }));

const NIM_BASE = 'https://integrate.api.nvidia.com/v1';

app.get('/', (req, res) => res.json({ status: 'ok', service: 'nim-proxy' }));

app.all('/v1/*', async (req, res) => {
  const secret = process.env.PROXY_SECRET;
  if (secret && req.headers['x-proxy-secret'] !== secret) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const nimPath = req.path.slice(3); // /v1/chat/completions → /chat/completions
  const url = `${NIM_BASE}${nimPath}`;

  // Override model if x-nim-model header is set
  const body = { ...req.body };
  if (req.headers['x-nim-model']) body.model = req.headers['x-nim-model'];

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
  console.log('NIM proxy listening on port', process.env.PORT || 3000)
);
