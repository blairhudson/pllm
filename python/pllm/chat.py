from __future__ import annotations

from typing import Any
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from pllm.runtime.client import OpenAI
from .settings import ClientSettings, config_path


def _choose_model(client: OpenAI, requested: str | None) -> str:
    if requested:
        return requested
    models = client.models.list().get("data", [])
    strict = [item["id"] for item in models if item.get("he", {}).get("privacy_mode") not in {"manifest_only", "trusted_backend"}]
    if len(strict) == 1:
        return strict[0]
    if not strict:
        raise RuntimeError("the server has no private models loaded")
    raise RuntimeError("more than one model is available; select one with --model: " + ", ".join(strict))


def run_chat(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    preparation_base_url: str | None = None,
    preparation_api_key: str | None = None,
    stream: bool = True,
    max_output_tokens: int = 64,
) -> None:
    settings = ClientSettings.load().merged(
        base_url=base_url,
        api_key=api_key,
        model=model,
        preparation_base_url=preparation_base_url,
        preparation_api_key=preparation_api_key,
    )
    console = Console()
    history_file = config_path().with_name("chat-history")
    history_file.parent.mkdir(parents=True, exist_ok=True)
    session: PromptSession[str] = PromptSession(history=FileHistory(str(history_file)))
    previous_response_id: str | None = None
    with OpenAI(base_url=settings.base_url, api_key=settings.api_key, default_model=settings.model,
                preparation_base_url=settings.preparation_base_url,
                preparation_api_key=settings.preparation_api_key,
                he_transport=settings.transport, correlation_mode=settings.correlation_mode,
                correlation_prefetch=settings.correlation_prefetch, token_cache_size=settings.token_cache_size,
                bundle_cache_mode=settings.bundle_cache_mode,
                bundle_cache_dir=settings.bundle_cache_dir,
                timeout=settings.timeout) as client:
        selected = _choose_model(client, settings.model)
        console.print(f"[bold]PLLM chat[/bold]  model={selected}  server={settings.base_url}")
        console.print("Commands: /clear /models /audit /config /help /quit")
        while True:
            try:
                prompt = session.prompt("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print(); return
            if not prompt: continue
            if prompt in {"/quit", "/exit"}: return
            if prompt == "/clear": previous_response_id = None; console.print("Conversation cleared."); continue
            if prompt == "/models": console.print_json(data=client.models.list()); continue
            if prompt == "/audit": console.print_json(data=client.privacy_audit.to_dict()); continue
            if prompt == "/config":
                console.print_json(data={"base_url": settings.base_url, "model": selected, "transport": settings.transport,
                                         "correlation_mode": settings.correlation_mode,
                                         "preparation_base_url": settings.preparation_base_url,
                                         "bundle_cache_mode": settings.bundle_cache_mode,
                                         "bundle_cache_dir": settings.bundle_cache_dir,
                                         "config_file": str(config_path())}); continue
            if prompt == "/help":
                console.print("/clear starts a new conversation. /audit shows the local privacy counters. /quit exits."); continue
            request: dict[str, Any] = {
                "model": selected,
                "input": prompt,
                "stream": stream,
                "max_output_tokens": max_output_tokens,
            }
            if previous_response_id: request["previous_response_id"] = previous_response_id
            if stream:
                final = None
                output_started = False
                status = console.status("[cyan]Running private inference...[/cyan]")
                status.start()
                try:
                    for event in client.responses.create(**request):
                        if event.type == "response.output_text.delta":
                            if not output_started:
                                status.stop()
                                console.print("[bold cyan]assistant>[/bold cyan] ", end="")
                                output_started = True
                            console.print(event.delta or "", end="", markup=False, highlight=False)
                        elif event.type == "response.completed": final = event.response
                finally:
                    status.stop()
                if not output_started:
                    console.print("[bold cyan]assistant>[/bold cyan] ", end="")
                console.print()
                if final: previous_response_id = str(final["id"])
            else:
                with console.status(
                    "[cyan]Running private inference...[/cyan]"
                ):
                    response = client.responses.create(**request)
                console.print(Markdown(response.output_text)); previous_response_id = response.id
