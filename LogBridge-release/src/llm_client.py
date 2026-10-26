"""Audited JSON calls; invoked only by explicit generate/judge commands."""
import hashlib
import json
import os
import time
import urllib.request


def call_json(prompt, model, role='generation', base_url=None):
    judge = role == 'judge'
    key_name = 'DEEPSEEK_API_KEY' if judge else 'OPENAI_API_KEY'
    key = os.environ.get(key_name)
    if not key:
        raise ValueError(f'{key_name} is required')
    base = (base_url or os.environ.get('DEEPSEEK_BASE_URL' if judge else 'OPENAI_BASE_URL')
            or ('https://api.deepseek.com/v1' if judge else 'https://api.openai.com/v1')).rstrip('/')
    if not base.endswith('/v1'):
        base += '/v1'
    body = {'model': model, 'messages': [{'role': 'system', 'content': 'Return valid JSON only.'},
                                        {'role': 'user', 'content': prompt}],
            'response_format': {'type': 'json_object'}}
    request = urllib.request.Request(base + '/chat/completions', data=json.dumps(body).encode(),
             headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    started = time.time()
    with urllib.request.urlopen(request, timeout=600) as response:
        envelope = json.load(response)
    content = envelope['choices'][0]['message']['content']
    parsed = json.loads(content)
    return parsed, {'requested_model': model, 'actual_model': envelope.get('model'),
                    'usage': envelope.get('usage', {}), 'elapsed_seconds': time.time()-started,
                    'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'response': envelope}
