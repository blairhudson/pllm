"""Stable CLI failure classes."""

from __future__ import annotations


class CLIError(Exception):
    def __init__(self, code: str, message: str, exit_code: int, stage: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code
        self.stage = stage


class UsageError(CLIError):
    def __init__(self, message: str) -> None:
        super().__init__("CLI_USAGE", message, 2, "parse")


class ResolutionError(CLIError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message, 3, "resolution")


class LocalIOError(CLIError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message, 4, "io")


class AuthorizationError(CLIError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message, 5, "authorization")


class RuntimeFailure(CLIError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message, 6, "runtime")


class PreparedMaterialError(CLIError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message, 7, "preparation")
