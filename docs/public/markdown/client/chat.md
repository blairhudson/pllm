# Chat from the terminal

Stream a conversation without writing application code.


```bash
pllm chat
```

The client uses the saved server, credential, and model. Override the connection for this invocation:

```bash
pllm chat --server http://127.0.0.1:8000 \
  --preparation-url http://127.0.0.1:8001 \
  --model private-model
```

## Commands

| Command | Behaviour |
| --- | --- |
| `/clear` | Start a new local conversation |
| `/models` | List the provider's models |
| `/audit` | Show local privacy counters |
| `/config` | Show active nonsecret connection settings |
| `/help` | Show available commands |
| `/quit` | Exit the client |

Use `--no-stream` to print a response after it completes. Each linear stage waits
for its preparation acknowledgement and masked inference result. Correction bytes
flow directly from preparation to inference and are reported separately.

## Local privacy

Terminal capture, shell sessions, and application logging can record plaintext even though the provider does not receive it. Run chat within your trusted environment. Avoid shared terminal recordings for sensitive prompts.

The audit counters report literal payload categories. A count of zero plaintext prompt bytes does not establish that all metadata is harmless. [Security boundaries](/docs/security) explains the activation scale issue and the remaining traffic signals.
