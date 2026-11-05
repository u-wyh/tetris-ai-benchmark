"""Bounded local Codex RPC for quota reads and an authorized stale-goal resume.

Never changes account settings, spends reset credits, or overrides a user pause.
"""

import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import time


class LocalRPC:
    def __init__(self, timeout=12):
        self.deadline = time.monotonic() + timeout
        self.buffer = b''
        self.sequence = 0
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.sock.settimeout(timeout)
            self.sock.connect(str(Path.home()/'.codex/app-server-control/app-server-control.sock'))
            key = base64.b64encode(os.urandom(16)).decode()
            self.sock.sendall(('GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n'
                               'Connection: Upgrade\r\nSec-WebSocket-Key: '+key+
                               '\r\nSec-WebSocket-Version: 13\r\n\r\n').encode())
            while b'\r\n\r\n' not in self.buffer:
                self.sock.settimeout(self.remaining())
                part = self.sock.recv(4096)
                if not part:
                    raise EOFError('Local Codex handshake closed')
                self.buffer += part
                if len(self.buffer) > 65536:
                    raise ValueError('Local Codex handshake too large')
            header, self.buffer = self.buffer.split(b'\r\n\r\n', 1)
            expected = base64.b64encode(hashlib.sha1((key+'258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest())
            if (b' 101 ' not in header.split(b'\r\n')[0]
                    or b'sec-websocket-accept: '+expected.lower() not in header.lower()):
                raise RuntimeError('Local Codex WebSocket handshake rejected')
            self.request('initialize', dict(clientInfo=dict(name='tetris-hourly-continuation', version='1.0'),
                                            capabilities=dict(experimentalApi=True)))
            self.send(dict(method='initialized', params={}))
        except BaseException:
            self.sock.close()
            raise

    def remaining(self):
        remaining = self.deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Local Codex RPC deadline exhausted')
        return remaining

    def take(self, size):
        while len(self.buffer) < size:
            self.sock.settimeout(self.remaining())
            part = self.sock.recv(max(4096, size-len(self.buffer)))
            if not part:
                raise EOFError('Local Codex connection closed')
            self.buffer += part
        result, self.buffer = self.buffer[:size], self.buffer[size:]
        return result

    def send(self, value, opcode=1):
        data = json.dumps(value).encode() if opcode == 1 else value
        size = len(data)
        header = (bytes([size | 128]) if size < 126 else
                  bytes([254])+struct.pack('!H', size) if size < 65536 else
                  bytes([255])+struct.pack('!Q', size))
        mask = os.urandom(4)
        self.sock.settimeout(self.remaining())
        self.sock.sendall(bytes([128 | opcode])+header+mask+
                          bytes(byte ^ mask[i % 4] for i, byte in enumerate(data)))

    def request(self, method, params):
        self.sequence += 1
        request_id = self.sequence
        self.send(dict(id=request_id, method=method, params=params))
        chunks = b''
        while True:
            first, second = self.take(2)
            size = second & 127
            if size == 126:
                size = struct.unpack('!H', self.take(2))[0]
            elif size == 127:
                size = struct.unpack('!Q', self.take(8))[0]
            if size + len(chunks) > 4*1024**2:
                raise ValueError('Local Codex message too large')
            mask = self.take(4) if second & 128 else None
            data = self.take(size)
            if mask:
                data = bytes(byte ^ mask[i % 4] for i, byte in enumerate(data))
            opcode = first & 15
            if opcode == 8:
                raise EOFError('Local Codex WebSocket closed')
            if opcode == 9:
                self.send(data, 10)
                continue
            if opcode == 10:
                continue
            if opcode not in (0, 1):
                raise ValueError('Unexpected local Codex frame')
            chunks += data
            if not first & 128:
                continue
            message = json.loads(chunks)
            chunks = b''
            if message.get('id') == request_id:
                if 'error' in message:
                    raise RuntimeError(f'Local Codex RPC {method} failed: {message["error"].get("code")}')
                return message['result']

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.sock.close()


def recover_available_goal(thread_id, client_factory=LocalRPC):
    """Resume only a usage-limited goal after the live account explicitly allows use."""
    limited = ('usageLimited', 'usage_limited')
    with client_factory() as client:
        before = client.request('thread/goal/get', dict(threadId=thread_id))['goal']
        if not before or before['status'] not in limited:
            return dict(ready=bool(before and before['status'] == 'active'), action='goal_changed')
        rates = client.request('account/rateLimits/read', dict(excludeResetCreditDetails=True))
        if rates.get('ordinaryUsageAllowed') is not True:
            return dict(ready=False, action='quota_unavailable')
        # A user pause, cancellation or budget change during the network read wins.
        before = client.request('thread/goal/get', dict(threadId=thread_id))['goal']
        if not before or before['status'] not in limited:
            return dict(ready=bool(before and before['status'] == 'active'), action='goal_changed')
        after = client.request('thread/goal/set', dict(threadId=thread_id, status='active'))['goal']
        if (after['objective'] != before['objective'] or after.get('tokenBudget') != before.get('tokenBudget')
                or after['tokensUsed'] < before['tokensUsed'] or after['status'] != 'active'):
            raise RuntimeError('Goal recovery failed to preserve objective, budget or usage')
        return dict(ready=True, action='quota_available_goal_resumed', goal_status='active',
                    ordinary_usage_allowed=True, tokens_used=after['tokensUsed'])
