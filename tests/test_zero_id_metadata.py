"""Ordinary metadata loader/scalar binder regression, not GKM source admission.

Run this file ALONE with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 and --noconftest.
No installed-wheel, historical GKM49, custody, or universal entropy claim.
"""
from __future__ import annotations

import builtins
from contextlib import contextmanager
from enum import Enum
from functools import partial
import hashlib
import importlib
import importlib.util
import json
import sys
from types import BuiltinFunctionType, CellType, FunctionType, MethodType, ModuleType

import pytest


_MISSING = object()
_MAX_NODES = 65536
_MAX_DEPTH = 24
_FRESH = (
    "pllm.runtime.transformer_engine", "pllm.runtime.model_binding",
    "pllm.runtime.transformer_client",
)
_BOUNDARIES = {
    "os": ("urandom", "system", "popen"),
    "socket": ("socket", "create_connection", "getaddrinfo"),
    "_socket": ("socket", "getaddrinfo"),
    "urllib.request": ("urlopen", "urlretrieve"),
    "http.client": ("HTTPConnection", "HTTPSConnection"),
    "subprocess": ("Popen", "run", "call", "check_call", "check_output"),
    "secrets": ("token_bytes", "token_hex", "token_urlsafe", "choice", "randbelow", "randbits"),
    "random": ("Random", "seed", "random", "getrandbits", "randbytes", "randrange",
               "randint", "choice", "choices", "shuffle", "sample", "uniform"),
    "numpy": ("ndarray", "array", "asarray", "ascontiguousarray", "empty", "empty_like", "zeros",
              "zeros_like", "ones", "ones_like", "full", "full_like", "arange", "frombuffer",
              "fromfile", "memmap", "load", "dot", "matmul", "einsum"),
    "pllm.model_loader": ("load_model", "resolve_model", "_resolve_model", "_source_files",
                          "_source_lock", "_file_digest"),
    "pllm.modeling": ("lower_model",),
    "pllm.compiler": ("compile", "compile_model", "compile_plan"),
    "pllm.runtime.model_binding": ("compile_runtime_model",),
    "pllm.runtime.loaders": ("load_hf_directory", "create_tiny_llama_checkpoint"),
    "pllm.runtime.hf_hub": ("resolve_huggingface_source",),
    "pllm.runtime.delegated_source": ("admit_source", "_admit_source", "_files", "_locked", "_tokenizer"),
}
_CLASSES = {
    "pllm.runtime.transformer_engine": {"MaskedTransformerEngine": ("__init__", "load")},
    "pllm.runtime.transformer_client": {"ClientBundle": ("tokenizer",)},
    "pllm.runtime.safetensors_store": {"SafeTensorStore": (
        "__init__", "get", "get_slice", "iter_slices", "get_linear", "_discover")},
}
_CONSTRUCTORS = {
    "MaskedGEMM", "CompiledMatrix", "PagedGEMM", "ModelByteTokenizer", "ByteTokenizer",
    "AlphabetTokenizer", "SentencePieceTokenizer", "TokenizersJSONTokenizer", "HFTokenizer",
    "GkmSourceBuilder", "GkmClientRequest", "GkmRoleRequest", "FssSourceBuilder",
    "FssClientRequest", "ReplicatedSourcePublisher", "FilePayload",
}
_EXTERNAL = ("tokenizers", "sentencepiece", "safetensors", "requests", "httpx",
             "huggingface_hub", "torch", "tenseal", "_sealapi_cpp")


def _assert_zero_id_ledger_restored(ledger):
    """Check every recorded attribute, hook, module and parent binding by identity."""
    for owner, key, old, mapping, _ in ledger["changes"]:
        namespace = owner if mapping else vars(owner)
        assert namespace.get(key, _MISSING) is old, key
    assert len(sys.meta_path) == len(ledger["meta_path_order"])
    assert all(current is original for current, original in
               zip(sys.meta_path, ledger["meta_path_order"]))
    for owner, key, _, original, mapping in ledger["retention_repairs"]:
        namespace = owner if mapping else vars(owner)
        assert namespace[key] is original, key
    for owner, _, originals in ledger["retention_sets"]:
        assert {id(item) for item in owner} == {id(item) for item in originals}
    for owner, key, _, original in ledger["retention_slots"]:
        if type(owner) is FunctionType:
            current = owner.__defaults__ if key == "__defaults__" else owner.__kwdefaults__
        else:
            assert type(owner) is CellType and key == "cell_contents"
            current = owner.cell_contents
        assert current is original, key
    for owner, _, original in ledger["retention_partials"]:
        assert owner.func is original[0]
        assert owner.args is original[1]
        assert owner.keywords is original[2]
        assert owner.__dict__ is original[3]
    assert ledger["restored"]


@contextmanager
def _zero_id_metadata_guard(monkeypatch, *, _fail_setup=False, _setup_import=None, _ledger=None):
    """Bounded import/API guard. No lazy np.random lookup or irreversible audit hook.

    Inspection covers bounded module namespaces and directly exported Python
    classes, exact builtin containers, Python function captures, exact method
    wrappers and partial slots. Each pass has fixed depth/node limits and an
    identity cycle set. Nested original aliases/captures refuse, never mutate.

    Module execution necessarily precedes post-load inspection. Dynamic lateral
    aliases, function globals, properties, opaque instance graphs and partial
    custom attributes are OUTSIDE this contract, not universal entropy proof.
    Unsupported callable/custom-container captures refuse without invocation.

    Fresh ordinary Python dependencies stay loaded. Supported retained guard
    aliases are repaired to their recorded original identities BEFORE patches
    unwind, including import/body/setup failure and validation refusal. Cleanup
    does NOT validate first: each known supported reference gets a repair attempt,
    then deferred failures report after hook/binding restoration. Fresh wrappers
    are rebuilt with a shared identity memo; their underlying callable identity
    and supported alias relationships, not universal wrapper identity, survive.
    Original deferred/active exceptions aggregate in order without flattening;
    a sole active exception propagates naturally. Only guard-owned pllm import
    bindings are removed.
    """
    assert not any(name in sys.modules for name in _FRESH), "run this file alone with --noconftest"
    ledger = {"changes": [], "meta_path_order": tuple(sys.meta_path), "restored": False,
              "descriptor_calls": 0, "control_calls": 0, "control": False,
              "retention_repairs": [], "retention_sets": [], "retention_slots": [],
              "retention_partials": [], "retention_errors": []}
    if _ledger is not None:
        assert type(_ledger) is dict and not _ledger
        _ledger.update(ledger)
        ledger = _ledger
    entry_modules = dict(sys.modules)
    fresh_dependencies = {}
    recorded = set()
    prohibited = {}
    hook_aliases = {}
    reverse = {}
    traps = {}
    wrapped_loaders = set()
    original_import = builtins.__import__
    original_import_module = importlib.import_module
    original_finders = tuple(sys.meta_path)

    def defer(error):
        if not any(error is previous for previous in ledger["retention_errors"]):
            ledger["retention_errors"].append(error)

    def finalize(active):
        errors = list(ledger["retention_errors"])
        if active is not None and not any(active is error for error in errors):
            errors.append(active)
        if not errors or len(errors) == 1 and errors[0] is active:
            return  # A sole active exception propagates naturally, without re-raise.
        if len(errors) == 1:
            raise errors[0]
        raise BaseExceptionGroup("metadata guard failures", errors)

    def record(owner, key, *, mapping=False, imported=False):
        identity = (id(owner), key, mapping)
        if identity not in recorded:
            recorded.add(identity)
            namespace = owner if mapping else vars(owner)
            ledger["changes"].append((owner, key, namespace.get(key, _MISSING), mapping, imported))

    def unwrap(value):
        return value.__func__ if type(value) in (staticmethod, classmethod) else value

    def register(value):
        value = unwrap(value)
        if id(value) not in reverse:
            prohibited[id(value)] = value
            if type(value) is MethodType:
                prohibited[id(value.__func__)] = value.__func__

    def trap_for(original):
        if id(original) not in traps:
            def refusal(*args, **kwargs):
                key = "control_calls" if ledger["control"] else "descriptor_calls"
                ledger[key] += 1
                raise AssertionError("metadata-only forbidden API boundary")

            traps[id(original)] = refusal
            reverse[id(refusal)] = (refusal, original)
        return traps[id(original)]

    def rng_import_boundary():
        """Identity only; never execute this sentinel as a substitute API."""

    import_refusal = trap_for(rng_import_boundary)

    def bounded(name):
        return (name == "pllm" or name.startswith("pllm.") or name in _BOUNDARIES
                or name == "numpy.random" or name.startswith("numpy.random.")
                or any(name == root or name.startswith(root + ".") for root in _EXTERNAL))

    def check_import(name):
        if name == "numpy.random" or name.startswith("numpy.random."):
            import_refusal()

    def absolute(name, globals=None, level=0):
        if not level:
            return name
        namespace = globals or {}
        package = namespace.get("__package__")
        if not package:
            package = namespace.get("__name__", "")
            if "__path__" not in namespace:
                package = package.rpartition(".")[0]
        return importlib.util.resolve_name("." * level + name, package)

    def is_prohibited(value):
        if id(value) in prohibited:
            return True
        return type(value) is MethodType and id(value.__func__) in prohibited

    def replacement(value):
        if id(value) in reverse:
            return None
        if type(value) in (staticmethod, classmethod):
            inner = replacement(value.__func__)
            if inner is not None:
                wrapped = type(value)(inner)
                reverse[id(wrapped)] = (wrapped, value)
                return wrapped
            return None
        if id(value) in hook_aliases:
            return hook_aliases[id(value)]
        return trap_for(value) if is_prohibited(value) else None

    def python_class(value, state):
        # No metaclass/instance/property lookup, and no traversal of native types.
        if type(value) is not type:
            return False
        if id(value) not in state["classes"]:
            consume(state, len(vars(value)))
            state["classes"][id(value)] = any(
                type(member) in (FunctionType, staticmethod, classmethod)
                for member in vars(value).values())
        return state["classes"][id(value)]

    def budget():
        return {"left": _MAX_NODES, "seen": set(), "classes": {}, "class_slots": set()}

    def consume(state, count=1):
        state["left"] -= count
        if state["left"] < 0:
            raise AssertionError("metadata guard traversal node budget exhausted")

    def custom_container(kind, state):
        # Read the known type-layout slot, not an instance's __class__ property
        # or a user metaclass's overridden attribute lookup.
        bases = vars(type)["__mro__"].__get__(kind)
        consume(state, len(bases))
        return any(base is container for base in bases
                   for container in (dict, list, tuple, set, frozenset))

    def visit(value, state, *, depth=0, capture=False, retaining=False):
        """Validate only approved shapes. Never call a captured callable."""
        if depth > _MAX_DEPTH:
            raise AssertionError("metadata guard traversal depth exhausted")
        consume(state)
        identity = (id(value), capture)
        if identity in state["seen"]:
            return
        state["seen"].add(identity)
        if id(value) in reverse:
            if capture:
                raise AssertionError("retained guard capture refused")
            return
        if type(value) is MethodType and id(value.__func__) in reverse:
            raise AssertionError("retained bound guard method refused")
        if is_prohibited(unwrap(value)) or id(unwrap(value)) in hook_aliases:
            if not retaining:
                raise AssertionError("unsafe nested original alias/capture")
            return
        kind = type(value)
        if kind in (dict, list, tuple, set, frozenset):
            if kind is dict and retaining:
                # Guard-valued dict keys are not a supported reversible slot.
                for key in value:
                    visit(key, state, depth=depth + 1, capture=True, retaining=True)
                children = value.values()
            else:
                children = (item for pair in value.items() for item in pair) if kind is dict else value
            for item in children:
                visit(item, state, depth=depth + 1, capture=capture, retaining=retaining)
        elif kind in (staticmethod, classmethod):
            visit(value.__func__, state, depth=depth + 1, capture=capture, retaining=retaining)
        elif kind is FunctionType:
            children = [value.__defaults__, value.__kwdefaults__]
            for cell in value.__closure__ or ():
                try:
                    children.append(cell.cell_contents)
                except ValueError:
                    pass
            for item in children:
                visit(item, state, depth=depth + 1, capture=True, retaining=retaining)
        elif kind is partial:
            for item in (value.func, value.args, value.keywords):
                visit(item, state, depth=depth + 1, capture=True, retaining=retaining)
        elif kind is CellType:
            try:
                item = value.cell_contents
            except ValueError:
                return
            visit(item, state, depth=depth + 1, capture=True, retaining=retaining)
        elif capture and (callable(value) and kind not in (BuiltinFunctionType, type)
                          or custom_container(kind, state)):
            raise AssertionError("unsupported callable/container capture")
        # Other noncallable opaque metadata is a leaf, not an instance graph.

    def slots(module, state):
        """Only module namespace plus directly exported ordinary Python classes."""
        namespace = vars(module)
        consume(state, len(namespace))
        for key, value in tuple(namespace.items()):
            if not key.startswith("__"):
                yield module, key, value
        for value in tuple(namespace.values()):
            if (python_class(value, state) and not is_prohibited(value)
                    and id(value) not in state["class_slots"]):
                state["class_slots"].add(id(value))
                for key, member in tuple(vars(value).items()):
                    if key not in {"__dict__", "__weakref__", "__builtins__"}:
                        yield value, key, member

    def exports(module):
        """Register ORIGINAL identities before changing any export or alias."""
        name = vars(module).get("__name__", "")
        selected = []
        namespace = vars(module)
        for key in _BOUNDARIES.get(name, ()):
            if callable(namespace.get(key)):
                selected.append((module, key, namespace[key]))
        if name.startswith("pllm."):
            for key in _CONSTRUCTORS:
                if callable(namespace.get(key)):
                    selected.append((module, key, namespace[key]))
        for cls_name, methods in _CLASSES.get(name, {}).items():
            cls = namespace.get(cls_name)
            if isinstance(cls, type):
                for key in methods:
                    if key in vars(cls):
                        selected.append((cls, key, vars(cls)[key]))
        native = name == "pllm._native" or name.startswith("pllm._native.")
        external = any(name == root or name.startswith(root + ".") for root in _EXTERNAL)
        rng = name == "numpy.random" or name.startswith("numpy.random.")
        for key, value in tuple(namespace.items()):
            if id(value) in reverse:
                continue
            if native:
                if key.startswith("__"):
                    continue
                if isinstance(value, type) and issubclass(value, (Enum, BaseException)):
                    continue  # Actual metadata enum/exception types, not factories.
                if type(value) in (str, bytes, int, float, bool, type(None)):
                    continue
                if isinstance(value, (BuiltinFunctionType, type)):
                    # All native constructors/entry points are outside this test.
                    selected.append((module, key, value))
                else:
                    raise AssertionError("unclassified native surface: " + name + "." + key)
            elif (external or rng) and callable(value) and not key.startswith("__"):
                if isinstance(value, type) and issubclass(value, BaseException):
                    continue
                selected.append((module, key, value))
        selected = [item for item in selected if id(item[2]) not in reverse]
        for _, _, value in selected:
            register(value)
        return selected

    def protect_loaded(scoped, ready=None):
        # Discover loaded RNG descendants through sys.modules ONLY.
        modules = []
        for name, module in tuple(sys.modules.items()):
            if not bounded(name) or not isinstance(module, ModuleType):
                continue
            spec = vars(module).get("__spec__")
            if module is not ready and spec is not None and vars(spec).get("_initializing", False):
                continue  # Bootstrap metadata is protected at loader completion.
            modules.append(module)
        if sum(len(vars(module)) for module in modules) > _MAX_NODES:
            raise AssertionError("metadata guard traversal node budget exhausted")
        selected = [item for module in modules for item in exports(module)]
        state = budget()
        for module in modules:
            for _, _, value in slots(module, state):
                if replacement(value) is None:
                    visit(value, state)
        for owner, key, value in selected:
            substitute = replacement(value)
            if substitute is not None:
                record(owner, key)
                scoped.setattr(owner, key, substitute)
        state = budget()
        for module in modules:
            for owner, key, value in slots(module, state):
                substitute = replacement(value)
                if substitute is not None:
                    record(owner, key)
                    scoped.setattr(owner, key, substitute)

    def track_fresh(module):
        namespace = vars(module)
        name = namespace.get("__name__", "")
        path = namespace.get("__file__", "")
        if (type(module) is ModuleType and entry_modules.get(name, _MISSING) is not module
                and type(path) is str and path.endswith((".py", ".pyc"))):
            fresh_dependencies[id(module)] = module

    def retention(*, repair=False):
        """Refuse on import; independently repair known shapes at unwind.

        Validation failure is sticky, not a reason to skip cleanup. Cleanup has
        per-root budgets, continues through sibling/module failures and reports
        errors only after all guard patches unwind. No opaque heap traversal.
        Mutable function/cell/partial slots keep identity; immutable wrappers and
        containers share one reconstruction memo across the supported graph.
        """
        for module in tuple(sys.modules.values()):
            if type(module) is ModuleType:
                track_fresh(module)
        if not repair:
            try:
                state = budget()
                for module in tuple(fresh_dependencies.values()):
                    for _, _, value in slots(module, state):
                        visit(value, state, retaining=True)
            except BaseException as error:
                defer(error)
                raise
            return
        state = budget()
        memo = {}
        active = []

        def safe_hash(value, depth=0):
            state["left"] -= 1
            if depth > _MAX_DEPTH or state["left"] < 0:
                raise AssertionError("metadata guard hash-shape budget exhausted")
            if type(value) in (str, bytes, int, float, bool, type(None),
                               FunctionType, BuiltinFunctionType, staticmethod,
                               classmethod, partial):
                return True
            return type(value) in (tuple, frozenset) and all(
                safe_hash(item, depth + 1) for item in value)

        def write(owner, key, old, original, *, mapping=True):
            if (original is not old and mapping and type(owner) is dict
                    and not safe_hash(key)):
                raise AssertionError("unrepairable opaque retention key")
            if original is not old:
                # This module/slot did not exist on entry. Both directions are
                # recorded; the desired original is exact reverse provenance,
                # never a guessed reconstruction of a prohibited callable.
                ledger["retention_repairs"].append((owner, key, old, original, mapping))
                if mapping:
                    owner[key] = original
                else:
                    setattr(owner, key, original)

        def write_slot(owner, key, old, original):
            # Exact Python function/cell C slots only, never user descriptors.
            if original is not old:
                ledger["retention_slots"].append((owner, key, old, original))
                setattr(owner, key, original)

        def rewrite(value, depth=0):
            try:
                return rewrite_known(value, depth)
            except BaseException as error:
                defer(error)
                return value  # Continue every other known sibling/root/module.

        def rewrite_known(value, depth):
            if depth > _MAX_DEPTH:
                raise AssertionError("metadata guard traversal depth exhausted")
            state["left"] -= 1
            if state["left"] < 0:
                raise AssertionError("metadata guard traversal node budget exhausted")
            if id(value) in reverse:
                return reverse[id(value)][1]
            for index, (identity, _) in enumerate(active):
                if identity == id(value):
                    if any(immutable for _, immutable in active[index:]):
                        raise AssertionError("unrepairable immutable retention cycle")
                    return value
            if id(value) in memo:
                return memo[id(value)]
            kind = type(value)
            supported = (dict, list, tuple, set, frozenset, FunctionType, CellType,
                         staticmethod, classmethod, partial, MethodType)
            if kind not in supported:
                return value  # Opaque instance graphs remain outside the contract.
            active.append((id(value), kind in (tuple, frozenset, staticmethod,
                                              classmethod, MethodType)))
            memo[id(value)] = value
            try:
                if kind is FunctionType:
                    for key, item in (("__defaults__", value.__defaults__),
                                      ("__kwdefaults__", value.__kwdefaults__)):
                        try:
                            write_slot(value, key, item, rewrite(item, depth + 1))
                        except BaseException as error:
                            defer(error)
                    for cell in value.__closure__ or ():
                        rewrite(cell, depth + 1)
                    result = value
                elif kind is CellType:
                    try:
                        item = value.cell_contents
                    except ValueError:
                        return value
                    write_slot(value, "cell_contents", item, rewrite(item, depth + 1))
                    result = value
                elif kind in (staticmethod, classmethod):
                    original = rewrite(value.__func__, depth + 1)
                    result = value if original is value.__func__ else kind(original)
                elif kind is MethodType:
                    original = rewrite(value.__func__, depth + 1)
                    result = (value if original is value.__func__ else
                              MethodType(original, value.__self__))
                elif kind is partial:
                    old = (value.func, value.args, value.keywords, value.__dict__)
                    original = (rewrite(old[0], depth + 1), rewrite(old[1], depth + 1),
                                rewrite(old[2], depth + 1), old[3])
                    if any(before is not after for before, after in zip(old, original)):
                        # functools.partial's exact C state API invokes no wrapped
                        # callable. In-place repair preserves partial, args/keyword
                        # aliases; custom attributes are preserved, NOT traversed.
                        ledger["retention_partials"].append((value, old, original))
                        partial.__setstate__(value, original)
                    result = value
                elif kind is dict:
                    for key, item in tuple(value.items()):
                        try:
                            if id(key) in reverse:
                                defer(AssertionError("unsupported retained guard dict key"))
                            write(value, key, item, rewrite(item, depth + 1))
                        except BaseException as error:
                            defer(error)
                    result = value
                elif kind is list:
                    for key, item in enumerate(tuple(value)):
                        try:
                            write(value, key, item, rewrite(item, depth + 1))
                        except BaseException as error:
                            defer(error)
                    result = value
                else:
                    originals = [rewrite(item, depth + 1) for item in value]
                    if all(old is new for old, new in zip(value, originals)):
                        result = value
                    elif kind is tuple:
                        result = tuple(originals)
                    else:
                        if not all(safe_hash(item) for item in (*value, *originals)):
                            raise AssertionError("unrepairable opaque retention member")
                        if kind is frozenset:
                            result = frozenset(originals)
                        else:
                            result = value
                        if kind is set:
                            ledger["retention_sets"].append((value, tuple(value), tuple(originals)))
                            value.clear()
                            value.update(originals)
                memo[id(value)] = result
                return result
            finally:
                active.pop()

        def cleanup_slots(module):
            # A rejected class/module must not starve another known namespace.
            namespace = vars(module)
            consume(budget(), len(namespace))
            for key, value in tuple(namespace.items()):
                if not key.startswith("__"):
                    yield module, key, value
            seen_classes = set()
            for value in tuple(namespace.values()):
                try:
                    if (id(value) not in seen_classes and python_class(value, budget())
                            and not is_prohibited(value)):
                        seen_classes.add(id(value))
                        for key, member in tuple(vars(value).items()):
                            if key not in {"__dict__", "__weakref__", "__builtins__"}:
                                yield value, key, member
                except BaseException as error:
                    defer(error)

        for module in tuple(fresh_dependencies.values()):
            try:
                for owner, key, value in cleanup_slots(module):
                    state = budget()
                    try:
                        write(owner, key, value, rewrite(value), mapping=False)
                    except BaseException as error:
                        defer(error)
            except BaseException as error:
                defer(error)

    def own_module(name):
        if (name == "pllm" or name.startswith("pllm.")) and name not in sys.modules:
            record(sys.modules, name, mapping=True, imported=True)
            parent, _, child = name.rpartition(".")
            module = sys.modules.get(parent)
            if module is not None:
                record(module, child, imported=True)

    # No production/dependency imports occur between these definitions and setup.
    escaping: BaseException | None = None
    try:
        with monkeypatch.context() as scoped:
            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                resolved = absolute(name, globals, level)
                check_import(resolved)
                for item in fromlist or ():
                    if item != "*":
                        check_import(resolved + "." + item)
                if bounded(resolved):
                    protect_loaded(scoped)  # Includes cached imports and renamed aliases.
                result = original_import(name, globals, locals, fromlist, level)
                if bounded(resolved):
                    protect_loaded(scoped)
                retention()
                return result

            def guarded_import_module(name, package=None):
                resolved = importlib.util.resolve_name(name, package) if name.startswith(".") else name
                check_import(resolved)
                if bounded(resolved):
                    protect_loaded(scoped)
                result = original_import_module(name, package)
                if bounded(resolved):
                    protect_loaded(scoped)
                retention()
                return result

            def wrap_loader(spec):
                delegate = spec.loader
                if delegate is None or id(delegate) in wrapped_loaders:
                    return
                wrapped_loaders.add(id(delegate))
                # Patch instance method slots, not spec.loader: no guard wrapper
                # leaks through a dependency's __loader__ or __spec__ after exit.
                namespace = vars(delegate)
                create = getattr(delegate, "create_module", None)
                execute = getattr(delegate, "exec_module", None)
                assert execute is not None, "unclassified metadata loader: " + spec.name

                def create_module(spec):
                    module = None if create is None else create(spec)
                    if module is not None:
                        # Extension create_module can populate exports before exec_module.
                        for owner, key, value in exports(module):
                            record(owner, key)
                            scoped.setattr(owner, key, replacement(value))
                    return module

                def exec_module(module):
                    track_fresh(module)
                    try:
                        execute(module)
                    except BaseException as error:
                        defer(error)
                        raise
                    finally:
                        if bounded(vars(module).get("__name__", "")):
                            protect_loaded(scoped, ready=module)
                        retention()

                for key, value in (("create_module", create_module), ("exec_module", exec_module)):
                    record(namespace, key, mapping=True)
                    scoped.setitem(namespace, key, value)

            class Finder:
                def find_spec(self, fullname, path=None, target=None):
                    check_import(fullname)
                    own_module(fullname)
                    for finder in original_finders:
                        spec = finder.find_spec(fullname, path, target)
                        if spec is not None:
                            origin = spec.origin
                            if (bounded(fullname) or type(origin) is str
                                    and origin.endswith((".py", ".pyc"))):
                                wrap_loader(spec)
                            return spec
                    return None

            hook_aliases[id(original_import)] = guarded_import
            hook_aliases[id(original_import_module)] = guarded_import_module
            reverse[id(guarded_import)] = (guarded_import, original_import)
            reverse[id(guarded_import_module)] = (guarded_import_module, original_import_module)
            record(builtins, "__import__")
            scoped.setattr(builtins, "__import__", guarded_import)
            record(importlib, "import_module")
            scoped.setattr(importlib, "import_module", guarded_import_module)
            record(sys, "meta_path")
            scoped.setattr(sys, "meta_path", [Finder(), *original_finders])
            try:
                protect_loaded(scoped)
                if _setup_import is not None:
                    guarded_import_module(_setup_import)
                if _fail_setup:
                    raise RuntimeError("setup sentinel")
                yield ledger
                assert ledger["descriptor_calls"] == 0
            except BaseException as error:
                defer(error)
                raise
            finally:
                # Repair while provenance and all guard patches are still live.
                # The context then undoes API/hook/loader changes in reverse order.
                try:
                    retention(repair=True)
                except BaseException as error:
                    defer(error)  # Explicit patches must unwind even on discovery failure.
    except BaseException as error:
        escaping = error
        defer(error)  # Includes monkeypatch.context unwind failures.
        raise
    finally:
        # monkeypatch.context already restored all explicit patches, even if setup
        # failed. Restore only guard-owned import bindings, never other dependencies.
        for owner, key, old, mapping, imported in reversed(ledger["changes"]):
            if not imported:
                continue
            try:
                namespace = owner if mapping else vars(owner)
                if old is _MISSING:
                    namespace.pop(key, None)
                else:
                    namespace[key] = old
            except BaseException as error:
                defer(error)
        ledger["restored"] = True
        try:
            _assert_zero_id_ledger_restored(ledger)
        except BaseException as error:
            ledger["restored"] = False
            defer(error)
        finalize(escaping)


def _zero_id_metadata_manifest(path):
    from pllm.runtime.models import ModelManifest

    return ModelManifest(
        id="zero-id-metadata-only", architecture="qwen2", source_format="hf",
        source=str(path), vocab_size=8, hidden_size=2, intermediate_size=2,
        num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
        head_dim=2, context_length=16, created_at=0,
        chat_template="MANIFEST {{ messages }}",
    )


@pytest.mark.parametrize("special,expected", [
    ({"bos_token_id": 0, "eos_token_id": 0}, (0, 0)),
    ({"bos_token_id": 0, "eos_token_id": [0, 1]}, (0, 0)),
    ({"bos_token_id": 0, "eos_token_id": [1, 0]}, (0, 1)),
    ({"bos_token_id": 3, "eos_token_id": 4}, (3, 4)),
    ({}, (2, 1)),
    ({"bos_token_id": None, "eos_token_id": 0}, (2, 0)),
    ({"bos_token_id": 0, "eos_token_id": None}, (0, 1)),
    ({"bos_token_id": None, "eos_token_id": None}, (2, 1)),
    ({"bos_token_id": 0, "eos_token_id": []}, (0, 1)),
    # Ordinary loader still coerces; this is NOT GKM's strict admission policy.
    ({"bos_token_id": "0", "eos_token_id": ["0", 1]}, (0, 0)),
    ({"bos_token_id": False, "eos_token_id": True}, (0, 1)),
    ({"bos_token_id": -1, "eos_token_id": 8}, (-1, 8)),
])
@pytest.mark.parametrize("options,template,bos_token,eos_token,add_bos", [
    ({}, "MANIFEST {{ messages }}", "", "", True),
    ({"chat_template": None}, "MANIFEST {{ messages }}", "", "", True),
    ({"chat_template": ""}, "MANIFEST {{ messages }}", "", "", True),
    ({"chat_template": "SOURCE {{ messages }}", "bos_token": "<zero>",
      "eos_token": "<end>", "add_bos_token": False,
      "bos_token_id": 6, "eos_token_id": 7},
     "SOURCE {{ messages }}", "<zero>", "<end>", False),
])
def test_zero_id_metadata_descriptor_and_digest(
    tmp_path, monkeypatch, special, expected, options, template, bos_token, eos_token, add_bos,
):
    """Literal expectations; real loader/digest; scalar binder is not list admission."""
    raw = b"opaque synthetic tokenizer bytes\x00not a tokenizer model"
    with _zero_id_metadata_guard(monkeypatch) as ledger:
        from pllm.runtime.model_binding import RuntimeBindingError, _tokenizer_digest
        from pllm.runtime.transformer_engine import MaskedTransformerEngine

        manifest = _zero_id_metadata_manifest(tmp_path)
        (tmp_path / "tokenizer.json").write_bytes(raw)
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(options), encoding="utf-8")
        config = {"vocab_size": 8, **special}
        descriptor = MaskedTransformerEngine._load_tokenizer_descriptor(tmp_path, manifest, config)
        expected_descriptor = {
            "type": "tokenizer_json", "model": raw, "vocab_size": 8,
            "bos_token_id": expected[0], "eos_token_id": expected[1],
            "chat_template": template, "bos_token": bos_token, "eos_token": eos_token,
            "add_bos_token": add_bos,
        }
        assert descriptor == expected_descriptor
        runtime = {"vocab_size": 8, "bos_token_id": expected[0], "eos_token_id": expected[1]}
        canonical = {**expected_descriptor, "model": {
            "$bytes_sha256": hashlib.sha256(raw).hexdigest(), "length": len(raw),
        }}
        expected_digest = hashlib.sha256(json.dumps(
            canonical, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        digest = _tokenizer_digest(descriptor, runtime)
        assert digest == expected_digest
        assert _tokenizer_digest(dict(reversed(list(descriptor.items()))), runtime) == digest
        again = MaskedTransformerEngine._load_tokenizer_descriptor(tmp_path, manifest, config)
        assert _tokenizer_digest(again, runtime) == digest
        nullable = {**runtime, **{key: None for key in ("bos_token_id", "eos_token_id")
                                 if key in special and special[key] is None}}
        assert _tokenizer_digest(descriptor, nullable) == digest
        for key, wanted in zip(("bos_token_id", "eos_token_id"), expected):
            with pytest.raises(RuntimeBindingError, match=key):
                _tokenizer_digest({**descriptor, key: wanted + 1}, runtime)
            for invalid in (False, "0", [0]):
                with pytest.raises(RuntimeBindingError, match="must be an integer"):
                    _tokenizer_digest({**descriptor, key: invalid}, runtime)
        for change in ({"chat_template": "DIFFERENT"}, {"model": raw + b"!"}):
            assert _tokenizer_digest({**descriptor, **change}, runtime) != digest
        with pytest.raises(RuntimeBindingError, match="vocabulary"):
            _tokenizer_digest({**descriptor, "vocab_size": 9}, runtime)
        with pytest.raises(RuntimeBindingError, match="unsupported tokenizer type"):
            _tokenizer_digest({**descriptor, "type": "alphabet"}, runtime)
        assert ledger["descriptor_calls"] == 0
    _assert_zero_id_ledger_restored(ledger)


def test_zero_id_metadata_digest_distinguishes_zero_from_defaults(tmp_path, monkeypatch):
    with _zero_id_metadata_guard(monkeypatch) as ledger:
        from pllm.runtime.model_binding import _tokenizer_digest
        from pllm.runtime.transformer_engine import MaskedTransformerEngine

        manifest = _zero_id_metadata_manifest(tmp_path)
        (tmp_path / "tokenizer.json").write_bytes(b"opaque")
        (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        digests = []
        for special, bos, eos in (({}, 2, 1),
                                  ({"bos_token_id": 0, "eos_token_id": 0}, 0, 0)):
            descriptor = MaskedTransformerEngine._load_tokenizer_descriptor(
                tmp_path, manifest, special)
            runtime = {"vocab_size": 8, "bos_token_id": bos, "eos_token_id": eos}
            assert (descriptor["bos_token_id"], descriptor["eos_token_id"]) == (bos, eos)
            digests.append(_tokenizer_digest(descriptor, runtime))
        assert digests[0] != digests[1]
        assert ledger["descriptor_calls"] == 0
    _assert_zero_id_ledger_restored(ledger)


def test_zero_id_metadata_guard_restores_after_exception(tmp_path, monkeypatch):
    """Functional negative controls, bounded refusal and exact identity restoration."""
    original_rng = vars(sys.modules["os"])["urandom"]
    original_import = builtins.__import__
    original_import_module = importlib.import_module
    dependency_text = (
        "from os import urandom as retained_api\n"
        "from builtins import __import__ as retained_import\n"
        "from importlib import import_module as retained_import_module\n"
        "retained = {'nested': [retained_api, (retained_api,)], "
        "'set': {retained_api}, 'frozen': frozenset((retained_api,)), "
        "'hooks': [retained_import, (retained_import_module,)]}\n"
        "shared_static = staticmethod(retained_api)\n"
        "shared_class = classmethod(retained_api)\n"
        "class Retained:\n"
        "    f = shared_static\n"
        "    g = shared_class\n"
        "wrapper_aliases = [shared_static, (shared_class,)]\n"
        "retained['wrappers'] = {shared_static, shared_class}\n"
    )
    names = ("_zero_id_body_dep", "_zero_id_setup_dep", "_zero_id_import_dep")
    for name in names:
        (tmp_path / (name + ".py")).write_text(dependency_text, encoding="utf-8")
    (tmp_path / "_zero_id_failed_parent.py").write_text(
        "import _zero_id_import_dep\nraise RuntimeError('import sentinel')\n", encoding="utf-8")
    mixed_names = []
    captures_text = (
        "shared_partial = partial(retained_api, retained_api, api=retained_api)\n"
        "partial_args_alias = shared_partial.args\n"
        "partial_keywords_alias = shared_partial.keywords\n"
        "partial_aliases = [shared_partial, (shared_partial,)]\n"
        "def defaults(api=retained_api, operation=shared_partial, nested=retained):\n"
        "    return api, operation, nested\n"
        "defaults_alias = defaults.__defaults__\n"
        "def keyword_defaults(*, api=retained_api, operation=shared_partial, nested=retained):\n"
        "    return api, operation, nested\n"
        "kwdefaults_alias = keyword_defaults.__kwdefaults__\n"
        "def make_closure(api, operation):\n"
        "    def captured():\n"
        "        return api, operation\n"
        "    return captured\n"
        "captured = make_closure(retained_api, shared_partial)\n"
        "closure_cells_alias = captured.__closure__\n"
        "bound = Retained.g\n"
        "bound_alias = bound\n"
    )
    opaque_text = (
        "opaque_calls = []\n"
        "class Opaque:\n"
        "    def __call__(self):\n"
        "        opaque_calls.append('must not execute')\n"
        "opaque = Opaque()\n"
        "def opaque_default(unknown=opaque, api=retained_api):\n"
        "    return unknown, api\n"
    )
    opaque_key_text = (
        "class OpaqueKey:\n"
        "    pass\n"
        "opaque_key = OpaqueKey()\n"
        "opaque_retention = {opaque_key: retained_api}\n"
    )
    for failure in ("body", "setup", "import", "opaque_key", "system_exit",
                    "keyboard_interrupt", "caught_refusal"):
        good_name = "_zero_id_mixed_" + failure + "_good"
        bad_name = "_zero_id_mixed_" + failure + "_rejected"
        mixed_names.append((failure, good_name, bad_name))
        (tmp_path / (good_name + ".py")).write_text(dependency_text, encoding="utf-8")
        text = ("import " + good_name + " as repairable\nimport sys\n"
                "from functools import partial\n" + dependency_text +
                "repairable.rejected = sys.modules[__name__]\n")
        if failure == "opaque_key":
            text += opaque_text + opaque_key_text + captures_text
        else:
            text += (opaque_text + captures_text if failure == "setup" else
                     captures_text + opaque_text)
        if failure == "import":
            text += (
                "original_runtime = RuntimeError('mixed import sentinel')\n"
                "explicit_cause = LookupError('mixed import cause')\n"
                "try:\n"
                "    raise original_runtime from explicit_cause\n"
                "except RuntimeError:\n"
                "    original_tail = original_runtime.__traceback__\n"
                "    raise\n"
            )
        (tmp_path / (bad_name + ".py")).write_text(text, encoding="utf-8")
    ambient_refusal_name = "_zero_id_ambient_caught_refusal_dep"
    (tmp_path / (ambient_refusal_name + ".py")).write_text(
        "from functools import partial\n" + dependency_text + captures_text + opaque_text,
        encoding="utf-8")

    def assert_retained_original(module):
        assert module.retained_api is original_rng
        assert module.retained["nested"][0] is original_rng
        assert module.retained["nested"][1][0] is original_rng
        assert next(iter(module.retained["set"])) is original_rng
        assert next(iter(module.retained["frozen"])) is original_rng
        assert module.retained_import is original_import
        assert module.retained_import_module is original_import_module
        assert module.retained["hooks"][0] is original_import
        assert module.retained["hooks"][1][0] is original_import_module
        static = vars(module.Retained)["f"]
        class_method = vars(module.Retained)["g"]
        assert type(static) is staticmethod and static.__func__ is original_rng
        assert type(class_method) is classmethod and class_method.__func__ is original_rng
        assert module.shared_static is static
        assert module.shared_class is class_method
        assert module.wrapper_aliases[0] is static
        assert module.wrapper_aliases[1][0] is class_method
        assert {id(item) for item in module.retained["wrappers"]} == {id(static), id(class_method)}

    def assert_capture_original(module):
        assert_retained_original(module)
        assert module.defaults.__defaults__ is module.defaults_alias
        assert module.defaults_alias[0] is original_rng
        assert module.defaults_alias[1] is module.shared_partial
        assert module.defaults_alias[2] is module.retained
        assert module.keyword_defaults.__kwdefaults__ is module.kwdefaults_alias
        assert module.kwdefaults_alias["api"] is original_rng
        assert module.kwdefaults_alias["operation"] is module.shared_partial
        assert module.kwdefaults_alias["nested"] is module.retained
        assert module.shared_partial.func is original_rng
        assert module.shared_partial.args is module.partial_args_alias
        assert module.partial_args_alias[0] is original_rng
        assert module.shared_partial.keywords is module.partial_keywords_alias
        assert module.partial_keywords_alias["api"] is original_rng
        assert module.partial_aliases[0] is module.shared_partial
        assert module.partial_aliases[1][0] is module.shared_partial
        assert module.captured.__closure__ is module.closure_cells_alias
        assert {id(cell.cell_contents) for cell in module.closure_cells_alias} == {
            id(original_rng), id(module.shared_partial)}
        assert module.bound is module.bound_alias
        assert module.bound.__func__ is original_rng
        assert module.bound.__self__ is module.Retained
        assert module.opaque_default.__defaults__[0] is module.opaque
        assert module.opaque_default.__defaults__[1] is original_rng
        assert module.opaque_calls == []  # Opaque object itself was never traversed/called.

    def assert_original_failure(error, original, cause, tail):
        assert error is original
        assert error.__cause__ is cause
        current = error.__traceback__
        assert current is not None
        while current.tb_next is not None:
            current = current.tb_next
        assert current is tail  # Original raise site survives, not a replacement traceback.

    with monkeypatch.context() as environment:
        environment.syspath_prepend(str(tmp_path))
        preloaded_dependency = ModuleType("_zero_id_preloaded_dep")
        preloaded_dependency.__file__ = str(tmp_path / "_zero_id_preloaded_dep.py")
        preloaded_dependency.retained = {"api": original_rng}
        preloaded_namespace = dict(vars(preloaded_dependency))
        environment.setitem(sys.modules, preloaded_dependency.__name__, preloaded_dependency)
        with pytest.raises(RuntimeError, match="restore sentinel"):
            with _zero_id_metadata_guard(monkeypatch) as ledger:
                ledger["control"] = True

                def blocked(call):
                    before = ledger["control_calls"]
                    with pytest.raises(AssertionError, match="forbidden API boundary"):
                        call()
                    assert ledger["control_calls"] == before + 1
                    assert ledger["descriptor_calls"] == 0

                cached = ModuleType("numpy.random._zero_id_cached")
                future = "numpy.random._zero_id_future"
                with monkeypatch.context() as controls:
                    controls.setitem(sys.modules, cached.__name__, cached)
                    blocked(lambda: importlib.import_module(cached.__name__))
                    blocked(lambda: importlib.import_module(future))
                    blocked(lambda: builtins.__import__("numpy", fromlist=("random",)))
                    blocked(lambda: builtins.__import__("numpy.random", fromlist=("_zero_id_future",)))
                    blocked(lambda: builtins.__import__(
                        "_zero_id_future", {"__package__": "numpy.random"}, level=1))
                    blocked(lambda: sys.meta_path[0].find_spec(future))

                    alias = ModuleType("pllm._zero_id_alias_control")
                    alias.renamed_boundary = original_rng
                    alias.import_hook = original_import
                    alias.module_hook = original_import_module

                    class MethodAliases:
                        prohibited_static = staticmethod(original_rng)
                        prohibited_class = classmethod(original_rng)
                        static_hook = staticmethod(original_import_module)
                        class_hook = classmethod(original_import)

                    alias.MethodAliases = MethodAliases
                    controls.setitem(sys.modules, alias.__name__, alias)
                    imported = importlib.import_module(alias.__name__)
                    blocked(lambda: imported.renamed_boundary(1))
                    blocked(lambda: imported.import_hook("numpy", fromlist=("random",)))
                    blocked(lambda: imported.module_hook(future))
                    blocked(lambda: MethodAliases.prohibited_static(1))
                    blocked(lambda: MethodAliases.prohibited_class(1))
                    blocked(lambda: MethodAliases.static_hook(future))
                    blocked(lambda: vars(MethodAliases)["class_hook"].__func__(
                        "numpy", fromlist=("random",)))
                    assert type(vars(MethodAliases)["prohibited_static"]) is staticmethod
                    assert type(vars(MethodAliases)["prohibited_class"]) is classmethod
                    assert vars(MethodAliases)["prohibited_static"].__func__ is imported.renamed_boundary
                    assert vars(MethodAliases)["prohibited_class"].__func__ is imported.renamed_boundary
                    assert imported.renamed_boundary is not imported.import_hook

                    # Nested originals refuse; shared containers remain untouched.
                    for value in (original_rng, original_import, original_import_module):
                        nested = {"api": [value, (value,)]}
                        alias.nested = nested
                        before = ledger["control_calls"]
                        with pytest.raises(AssertionError, match="unsafe nested original"):
                            importlib.import_module(alias.__name__)
                        assert alias.nested is nested and nested["api"][0] is value
                        assert nested["api"][1][0] is value
                        assert ledger["control_calls"] == before
                        del alias.nested

                    calls = []

                    class UnsupportedCallable:
                        def __call__(self):
                            calls.append("must not execute")

                    class UnsupportedContainer(list):
                        pass

                    def closure(value):
                        def captured():
                            return value
                        return captured

                    for value, message in (
                        (partial(original_rng, 1), "unsafe nested original"),
                        (partial(original_import_module, future), "unsafe nested original"),
                        (original_import, "unsafe nested original"),
                        (UnsupportedCallable(), "unsupported callable/container capture"),
                        (partial(UnsupportedCallable()), "unsupported callable/container capture"),
                        (UnsupportedContainer([original_rng]), "unsupported callable/container capture"),
                    ):
                        def default(captured=value):
                            return captured

                        for function in (default, closure(value)):
                            alias.unsafe_capture = function
                            before = ledger["control_calls"]
                            with pytest.raises(AssertionError, match=message):
                                importlib.import_module(alias.__name__)
                            assert ledger["control_calls"] == before
                            assert calls == []
                        del alias.unsafe_capture

                    # Fixed-depth and identity-cycle behavior, without API calls.
                    cycle = []
                    cycle.append(cycle)
                    alias.supported_cycle = cycle
                    assert importlib.import_module(alias.__name__) is alias
                    del alias.supported_cycle
                    deep = []
                    for _ in range(_MAX_DEPTH + 2):
                        deep = [deep]
                    alias.too_deep = deep
                    with pytest.raises(AssertionError, match="depth exhausted"):
                        importlib.import_module(alias.__name__)
                    del alias.too_deep
                    alias.too_wide = [None] * (_MAX_NODES + 1)
                    with pytest.raises(AssertionError, match="node budget exhausted"):
                        importlib.import_module(alias.__name__)
                    del alias.too_wide

                from pllm.runtime.safetensors_store import SafeTensorStore
                from pllm.runtime.transformer_client import ClientBundle
                from pllm.runtime.bundle_storage import FilePayload
                from pllm import _native

                # These call patched production boundaries, never their bodies.
                assert len({id(SafeTensorStore.__init__), id(ClientBundle.tokenizer),
                            id(_native.Matrix), id(FilePayload)}) == 4
                blocked(lambda: SafeTensorStore.__init__(None))
                blocked(lambda: ClientBundle.tokenizer(None))
                blocked(lambda: _native.Matrix())
                blocked(lambda: FilePayload(()))

                fresh = importlib.import_module(names[0])
                assert fresh.retained_api is not original_rng
                blocked(lambda: fresh.retained["nested"][0](1))
                blocked(lambda: fresh.retained_import("numpy", fromlist=("random",)))
                blocked(lambda: fresh.Retained.f(1))
                blocked(lambda: fresh.Retained.g(1))
                ledger["control"] = False
                raise RuntimeError("restore sentinel")
        _assert_zero_id_ledger_restored(ledger)
        assert sys.modules[names[0]] is fresh  # No deletion/reload of the dependency.
        assert_retained_original(fresh)

        with pytest.raises(RuntimeError, match="setup sentinel"):
            with _zero_id_metadata_guard(monkeypatch, _setup_import=names[1], _fail_setup=True):
                pytest.fail("failed setup must not yield")
        assert_retained_original(sys.modules[names[1]])
        with pytest.raises(RuntimeError, match="import sentinel"):
            with _zero_id_metadata_guard(monkeypatch, _setup_import="_zero_id_failed_parent"):
                pytest.fail("failed import must not yield")
        assert_retained_original(sys.modules[names[2]])
        # One rejected dependency must not starve its repairable sibling. Both
        # remain inspectable after failure; no dependency deletion/reload here.
        for failure, good_name, bad_name in mixed_names:
            failed_ledger = {}
            capture_message = (
                "unsupported callable/container capture" if failure in ("setup", "opaque_key")
                else "retained guard capture refused")
            singleton = failure in ("setup", "caught_refusal")
            expected_type = (AssertionError if singleton else BaseExceptionGroup
                             if failure in ("system_exit", "keyboard_interrupt") else ExceptionGroup)
            original_active = None
            original_cause = LookupError("mixed body cause")
            original_tail = None
            captured_failure = None
            if failure in ("body", "system_exit", "keyboard_interrupt"):
                error_type = {"body": RuntimeError, "system_exit": SystemExit,
                              "keyboard_interrupt": KeyboardInterrupt}[failure]
                original_active = error_type("mixed " + failure + " sentinel")
            with pytest.raises(expected_type) as refused:
                setup = bad_name if failure in ("setup", "import", "opaque_key") else None
                with _zero_id_metadata_guard(
                    monkeypatch, _setup_import=setup, _ledger=failed_ledger,
                ):
                    if setup is not None:
                        pytest.fail("rejected setup/import must not yield")
                    try:
                        importlib.import_module(bad_name)
                    except AssertionError as error:
                        captured_failure = error
                        if original_active is not None:
                            try:
                                raise original_active from original_cause
                            except BaseException:
                                original_tail = original_active.__traceback__
                                raise
                        assert failure == "caught_refusal"
                    else:
                        pytest.fail("retained capture must refuse before import returns")
            _assert_zero_id_ledger_restored(failed_ledger)
            capture = next(error for error in failed_ledger["retention_errors"]
                           if type(error) is AssertionError and str(error) == capture_message)
            if captured_failure is not None:
                assert capture is captured_failure
            repairable = sys.modules[good_name]
            rejected = repairable.rejected  # Failed import's ordinary module object.
            assert rejected.__name__ == bad_name
            assert_retained_original(repairable)
            assert_capture_original(rejected)
            if singleton:
                expected = (capture,)
                assert refused.value is capture
            else:
                assert type(refused.value) is expected_type
                assert refused.value.message == "metadata guard failures"
                if failure == "opaque_key":
                    # Real cleanup write failure, not a synthetic deferred callback.
                    key_error = next(error for error in failed_ledger["retention_errors"]
                                     if type(error) is AssertionError and
                                     str(error) == "unrepairable opaque retention key")
                    expected = (capture, key_error)
                    assert next(iter(rejected.opaque_retention.values())) is not original_rng
                elif failure == "import":
                    original_active = rejected.original_runtime
                    original_cause = rejected.explicit_cause
                    original_tail = rejected.original_tail
                    expected = (original_active, capture)
                else:
                    expected = (capture, original_active)
                assert len(refused.value.exceptions) == len(expected)
                assert all(actual is original for actual, original in
                           zip(refused.value.exceptions, expected))
            assert len(failed_ledger["retention_errors"]) == len(expected)
            assert all(actual is original for actual, original in
                       zip(failed_ledger["retention_errors"], expected))
            if original_active is not None:
                member = refused.value.exceptions[0 if failure == "import" else 1]
                assert_original_failure(member, original_active, original_cause, original_tail)

        # A sole active BaseException is not swallowed or needlessly grouped.
        for error_type in (SystemExit, KeyboardInterrupt):
            signal = error_type("singleton active signal")
            with pytest.raises(error_type) as propagated:
                with _zero_id_metadata_guard(monkeypatch):
                    raise signal
            assert propagated.value is signal

        # Caller-handled exceptions are not escaping guard failures. Natural
        # __context__ links may remain; ambient errors must never become members.
        ambient = LookupError("unrelated caller handling refusal")
        original_refusal = None
        try:
            raise ambient
        except LookupError:
            with pytest.raises(AssertionError, match="retained guard capture refused") as refused:
                with _zero_id_metadata_guard(monkeypatch) as ambient_ledger:
                    try:
                        importlib.import_module(ambient_refusal_name)
                    except AssertionError as error:
                        original_refusal = error
                    else:
                        pytest.fail("retained capture must refuse before import returns")
            assert refused.value is original_refusal
            assert len(ambient_ledger["retention_errors"]) == 1
            assert ambient_ledger["retention_errors"][0] is original_refusal
            assert original_refusal is not ambient
            _assert_zero_id_ledger_restored(ambient_ledger)
            assert sys.exception() is ambient

        ambient = LookupError("unrelated caller handling normal exit")
        try:
            raise ambient
        except LookupError:
            with _zero_id_metadata_guard(monkeypatch) as normal_ledger:
                pass
            assert normal_ledger["retention_errors"] == []
            _assert_zero_id_ledger_restored(normal_ledger)
            assert sys.exception() is ambient

        # Actual yield-body StopIteration exercises contextlib/PEP 479 handling;
        # neither it nor RuntimeError becomes a group merely due to caller state.
        for error_type in (StopIteration, RuntimeError):
            ambient = LookupError("unrelated caller handling singleton")
            original = error_type("singleton yield-body failure")
            cause = ValueError("explicit singleton cause")
            tail = None
            original_context = None
            try:
                raise ambient
            except LookupError:
                with pytest.raises(error_type) as propagated:
                    with _zero_id_metadata_guard(monkeypatch) as singleton_ledger:
                        try:
                            raise original from cause
                        except (StopIteration, RuntimeError):
                            tail = original.__traceback__
                            original_context = original.__context__
                            raise
                assert_original_failure(propagated.value, original, cause, tail)
                assert original_context is ambient
                assert propagated.value.__context__ is original_context
                assert len(singleton_ledger["retention_errors"]) == 1
                assert singleton_ledger["retention_errors"][0] is original
                _assert_zero_id_ledger_restored(singleton_ledger)
                assert sys.exception() is ambient
        assert sys.modules[preloaded_dependency.__name__] is preloaded_dependency
        assert set(vars(preloaded_dependency)) == set(preloaded_namespace)
        for key, value in preloaded_namespace.items():
            assert vars(preloaded_dependency)[key] is value
        assert preloaded_dependency.retained["api"] is original_rng

        preloaded = ModuleType(_FRESH[0])
        environment.setitem(sys.modules, _FRESH[0], preloaded)
        with pytest.raises(AssertionError, match="run this file alone"):
            with _zero_id_metadata_guard(monkeypatch):
                pytest.fail("preloaded target must not yield")
        assert sys.modules[_FRESH[0]] is preloaded
