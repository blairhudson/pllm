"""Source-locked local-clear recurrent ownership probe; never a serving benchmark."""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "docs/evidence/research-constant-state-sources"
UPSTREAM = "536ecc007387a50e77603bb5d92100e9b07514cc"
MODELS = (
    ("tiiuae/falcon-mamba-7b", "080ad94b3619e2c2d0afa59bafdc6113465b7006", "08ec4cda33510a27a3a8cd760d27086bbab0c27a3773583f17f9b61bb8016298"),
    ("RWKV/rwkv-4-169m-pile", "46bdc280eb97b6141d5d51a935e0c4870ecaefcc", "2bc1d10594c235cf3e6553a48d9d25a1ee2dfa8b21c8dc860a3eabf4cd9d40ff"),
    ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"),
)
SOURCE_HASHES = {
    "falcon_mamba": "709d013910967f45b9456935f34970e52aa48e935569a814a660a355bec2e4a1",
    "rwkv": "4a3e4ce930652de46a7fdffb404930bb2389533b64de98126c9e227739582338",
    "qwen3_5": "496958ed2dc0bbf3d2ca0c0368cee9129742f08a79104c0951f505ac29d5a4e0",
}


class NumericMismatch(AssertionError):
    """Exact-copy-reference contract failed; never an unchanged result."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    paths = [Path(__file__), ROOT/"scripts/frontier_probe_support.py",
             ROOT/"tests/test_constant_state_probe.py",
             ROOT/"python/pllm/runtime/semantic_hybrid.py",
             ROOT/"docs/evidence/research-constant-state-audit.json"]
    paths.extend(SOURCE_DIR/f"{family}-config.json" for family in SOURCE_HASHES)
    paths.extend(SOURCE_DIR/f"modeling_{family}.py" for family in SOURCE_HASHES)
    identity = {str(path.resolve()):digest(path) for path in paths}
    for (_,_,expected),family in zip(MODELS,SOURCE_HASHES):
        if identity[str((SOURCE_DIR/f"{family}-config.json").resolve())] != expected or identity[str((SOURCE_DIR/f"modeling_{family}.py").resolve())] != SOURCE_HASHES[family]:
            raise ValueError("pinned config/oracle identity mismatch before work")
    return identity


def verify_identity(expected):
    if source_identity() != expected:
        raise ValueError("source identity changed during checks/workers")


@contextmanager
def frozen_sources(expected=None):
    before = source_identity()
    if expected is not None and before != expected:
        raise ValueError("worker source identity differs from coordinator freeze")
    try:
        yield before
    finally:
        verify_identity(before)


def commitment(array):
    """Exact-bit commitment to public synthetic arrays only, not private data."""
    array = np.ascontiguousarray(array)
    header = json.dumps({"dtype":array.dtype.str,"shape":list(array.shape)},sort_keys=True).encode()
    return hashlib.sha256(header+b"\0"+array.tobytes()).hexdigest()


def exact_bits(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not np.array_equal(actual.view(np.uint32),expected.view(np.uint32)):
        raise NumericMismatch("owned recurrence differs from copy_reference exact bits")


def locked_bytes(url, expected, path, fetch=False):
    if fetch and not path.exists():
        data = urllib.request.urlopen(url, timeout=30).read(1 << 20)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("upstream source identity changed")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("source lock mismatch")
    return data


def audit(fetch=False):
    models = []
    for (repo, revision, expected), family in zip(MODELS, SOURCE_HASHES):
        url = f"https://huggingface.co/{repo}/resolve/{revision}/config.json"
        config = json.loads(locked_bytes(url, expected, SOURCE_DIR / f"{family}-config.json", fetch))
        source_url = f"https://raw.githubusercontent.com/huggingface/transformers/{UPSTREAM}/src/transformers/models/{family}/modeling_{family}.py"
        locked_bytes(source_url, SOURCE_HASHES[family], SOURCE_DIR / f"modeling_{family}.py", fetch)
        c = config.get("text_config", config)
        if family == "falcon_mamba":
            # intermediate_size, NOT hidden_size * misleading legacy expand=16.
            state = c["num_hidden_layers"] * c["intermediate_size"] * (c["state_size"] + c["conv_kernel"]) * 4
            kv_per_token = 0
            context = None
        elif family == "rwkv":
            state = c["num_hidden_layers"] * (2 * c["hidden_size"] + 3 * c["attention_hidden_size"]) * 4
            kv_per_token = 0
            context = c["context_length"]
        else:
            recurrent_layers = c["layer_types"].count("linear_attention")
            channels = 2*c["linear_num_key_heads"]*c["linear_key_head_dim"] + c["linear_num_value_heads"]*c["linear_value_head_dim"]
            state = recurrent_layers * (c["linear_num_value_heads"]*c["linear_key_head_dim"]*c["linear_value_head_dim"] + channels*c["linear_conv_kernel_dim"]) * 4
            kv_per_token = c["layer_types"].count("full_attention") * 2*c["num_key_value_heads"]*c["head_dim"]*4
            context = c["max_position_embeddings"]
        cache = Path.home()/".cache/huggingface/hub"/("models--"+repo.replace("/", "--"))/"snapshots"/revision
        # A cache path alone is not evidence of complete weights.
        files = list(cache.glob("*.safetensors")) if cache.exists() else []
        models.append({"repo": repo, "revision": revision, "config_sha256": expected,
            "source_revision": UPSTREAM, "source_sha256": SOURCE_HASHES[family],
            "config_url": url, "source_url": source_url,
            "float32_retained_state_bytes": state, "float32_kv_bytes_per_token": kv_per_token,
            "configured_context": context, "cached_safetensors_files": len(files),
            "cached_safetensors_bytes": sum(p.stat().st_size for p in files),
            "full_checkpoint_import_validated": False})
    return {"scope": "pinned configuration/source audit, not model execution", "models": models}


def reference_class():
    path = ROOT/"python/pllm/runtime/semantic_hybrid.py"
    spec = importlib.util.spec_from_file_location("constant_state_existing_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # Execute the bytes whose hash was captured, not potentially stale pycache.
    source = path.read_bytes()
    module.__source_sha256__ = hashlib.sha256(source).hexdigest()
    exec(compile(source,str(path),"exec"),module.__dict__)
    if module.__source_sha256__ != digest(path):
        raise ValueError("actual imported semantic_hybrid bytes changed")
    return module.BoundedGatedDeltaRecurrence


class OwnedGatedDelta:
    """Research-only one-owner decode, bounded scratch, borrowed output.

    Inputs are local-clear float32 projected tensors, not protected shares.
    Invalid input fails before mutation. Numeric failure burns owner and zeros
    state. No state export on each step; explicit snapshot copies for validation.
    """
    def __init__(self, key_heads, value_heads, key_dim, value_dim):
        self.geometry = reference_class()(key_heads, value_heads, key_dim, value_dim)
        self.state = self.geometry.initial_state()
        self.q = np.empty((value_heads, key_dim), np.float32)
        self.k = np.empty_like(self.q)
        self.delta = np.empty((value_heads, value_dim), np.float32)
        self.output = np.empty_like(self.delta)
        self.outer = np.empty_like(self.state[0])
        self.active = True
        self.position = 0

    @property
    def owned_bytes(self):
        return sum(x.nbytes for x in (self.state, self.q, self.k, self.delta, self.output, self.outer))

    def step(self, q, k, v, g, beta):
        h = self.geometry
        shapes = ((1,1,h.key_heads,h.key_head_dim),)*2 + ((1,1,h.value_heads,h.value_head_dim),) + ((1,1,h.value_heads),)*2
        if not self.active or self.position >= 4096:
            raise ValueError("owner inactive or 4096-step research horizon exhausted")
        if any(not isinstance(x, np.ndarray) or x.dtype != np.float32 or x.shape != s or not np.all(np.isfinite(x)) for x,s in zip((q,k,v,g,beta),shapes)) or np.any(g>0) or np.any(beta<0) or np.any(beta>1):
            raise ValueError("invalid recurrence input")
        # Expanding grouped heads only fills fixed scratch, never grows state.
        for dst, src in ((self.q, q), (self.k, k)):
            normalized = src[0,0] / np.sqrt(np.sum(src[0,0]*src[0,0], axis=-1, keepdims=True)+np.float32(1e-6))
            dst.reshape(h.key_heads, h.value_heads//h.key_heads, h.key_head_dim)[:] = normalized[:,None,:]
        self.q *= np.float32(1/math.sqrt(h.key_head_dim))
        with np.errstate(over="ignore", invalid="ignore"):
            self.state[0] *= np.exp(g[0,0])[:,None,None]
            np.einsum("hkv,hk->hv",self.state[0],self.k,out=self.delta)
            np.subtract(v[0,0],self.delta,out=self.delta)
            self.delta *= beta[0,0,:,None]
            np.multiply(self.k[:,:,None],self.delta[:,None,:],out=self.outer)
            self.state[0] += self.outer
            np.einsum("hkv,hk->hv",self.state[0],self.q,out=self.output)
        if not np.all(np.isfinite(self.state)) or not np.all(np.isfinite(self.output)):
            self.active = False
            self.state.fill(0)
            raise ValueError("numeric domain exhausted; owner burned")
        self.position += 1
        return self.output[None,None]


class StableWeightedExpState:
    """Independent reusable log-scaled weighted exponential recurrence.

    This is ONLY RWKV-v4's projected WKV numeric contract, not its complete
    attention block, token-shift, channel mixer, decoder or private protocol.
    State is (numerator, denominator, log_scale), each float32 [channels].
    """
    def __init__(self, log_decay, first):
        if not isinstance(log_decay,np.ndarray) or log_decay.dtype != np.float32 or log_decay.ndim != 1 or not 1 <= log_decay.size <= 4096 or first.shape != log_decay.shape or first.dtype != np.float32 or not np.all(np.isfinite(log_decay)) or not np.all(np.isfinite(first)):
            raise ValueError("invalid weighted-exp geometry or coefficients")
        self.decay = -np.exp(log_decay)
        if not np.all(np.isfinite(self.decay)):
            raise ValueError("weighted-exp decay outside finite domain")
        self.first = first.copy()
        self.state = np.zeros((3,log_decay.size),np.float32)
        self.state[2].fill(np.float32(-1e38))
        self.position = 0

    def step(self, key, value):
        shape = self.first.shape
        if self.position >= 4096 or any(x.shape != shape or x.dtype != np.float32 or not np.all(np.isfinite(x)) for x in (key,value)):
            raise ValueError("invalid weighted-exp input or horizon")
        numerator,denominator,scale = self.state
        out_scale = np.maximum(scale,key+self.first)
        old = np.exp(scale-out_scale)
        new = np.exp(key+self.first-out_scale)
        output = (old*numerator+new*value)/(old*denominator+new)
        next_scale = np.maximum(scale+self.decay,key)
        old = np.exp(scale+self.decay-next_scale)
        new = np.exp(key-next_scale)
        next_state = np.stack((old*numerator+new*value,old*denominator+new,next_scale))
        if not np.all(np.isfinite(output)) or not np.all(np.isfinite(next_state)):
            raise ValueError("weighted-exp finite domain exceeded")
        self.state[:] = next_state
        self.position += 1
        return output


def inputs(key_heads=2, value_heads=4, key_dim=8, value_dim=6):
    # Identical public synthetic arithmetic in fresh control/candidate workers.
    # No prompts, token IDs, private values or random seeds in evidence.
    def wave(shape, offset):
        return np.sin(np.arange(math.prod(shape),dtype=np.float32)*np.float32(.37)+np.float32(offset)).reshape(shape)
    shape = (1,1,key_heads,key_dim)
    heads = (1,1,value_heads)
    return (wave(shape,.1),wave(shape,.7),wave((1,1,value_heads,value_dim),1.1),
            -np.abs(wave(heads,.3))-np.float32(.1),(wave(heads,.9)+np.float32(1))*np.float32(.5))


def upstream_oracle(q,k,v,g,beta,state):
    """Execute only AST-selected upstream oracle, never upstream runtime classes."""
    import torch
    path = SOURCE_DIR/"modeling_qwen3_5.py"
    locked_bytes("", SOURCE_HASHES["qwen3_5"], path)
    tree = ast.parse(path.read_text())
    names = {"l2norm", "torch_recurrent_gated_delta_rule"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != names:
        raise ValueError("oracle source contract missing")
    for node in selected:
        # Select pure fallback only: never invoke hub/kernel decorators.
        node.decorator_list = []
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),"exec"), namespace)
    repeat = v.shape[2]//q.shape[2]
    tensors = [torch.from_numpy(x.copy()) for x in (np.repeat(q,repeat,axis=2), np.repeat(k,repeat,axis=2),v,g,beta,state)]
    with torch.no_grad():
        output, final = namespace["torch_recurrent_gated_delta_rule"](*tensors[:5],initial_state=tensors[5],output_final_state=True,use_qk_l2norm_in_kernel=True)
    return output.numpy(),final.numpy(),torch.__version__


def weighted_exp_oracle():
    import torch
    path = SOURCE_DIR/"modeling_rwkv.py"
    locked_bytes("",SOURCE_HASHES["rwkv"],path)
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="rwkv_linear_attention_cpu")
    node.decorator_list = []
    namespace = {"torch":torch}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),namespace)
    decay = np.linspace(-3,1,8,dtype=np.float32)
    first = np.linspace(-.5,.5,8,dtype=np.float32)
    owner = StableWeightedExpState(decay,first)
    key = np.sin(np.arange(23*8,dtype=np.float32)).reshape(1,23,8)*np.float32(40)
    value = np.cos(key)
    actual = np.stack([owner.step(k,v) for k,v in zip(key[0],value[0])])[None]
    with torch.no_grad():
        expected,final = namespace["rwkv_linear_attention_cpu"](*(torch.from_numpy(x.copy()) for x in (decay,first,key,value)),return_state=True)
    np.testing.assert_allclose(actual,expected.numpy(),rtol=2e-5,atol=2e-6)
    expected_state = np.stack([x.numpy()[0] for x in final])
    np.testing.assert_allclose(owner.state,expected_state,rtol=2e-5,atol=2e-6)
    return {"checks":23,"channels":8,"output_max_absolute_error":float(np.max(np.abs(actual-expected.numpy()))),"state_max_absolute_error":float(np.max(np.abs(owner.state-expected_state)))}


def functional(oracle=False, geometry=(2,4,8,6), steps=17):
    with frozen_sources() as identity:
        result = _functional(oracle,geometry,steps)
        result["source_identity"] = identity
        return result


def _functional(oracle=False, geometry=(2,4,8,6), steps=17):
    owner = OwnedGatedDelta(*geometry)
    ref = owner.geometry
    state = ref.initial_state()
    err = 0.0
    torch_version = None
    for _ in range(steps):
        x = inputs(*geometry)
        prior = state.copy()
        expected,state = ref.evaluate(*x,state,mode="decode")
        actual = owner.step(*x)
        try:
            exact_bits(actual,expected)
            exact_bits(owner.state,state)
            if oracle:
                out,final,torch_version = upstream_oracle(*x,prior)
                np.testing.assert_allclose(actual,out,rtol=2e-5,atol=2e-6)
                np.testing.assert_allclose(owner.state,final,rtol=2e-5,atol=2e-6)
                err = max(err,float(np.max(np.abs(actual-out))),float(np.max(np.abs(owner.state-final))))
        except Exception:
            owner.active = False
            owner.state.fill(0)
            raise
    return {"checks":steps,"geometry":list(geometry),"copy_reference_exact_bits":True,"upstream_oracle":oracle,"upstream_max_absolute_error":err if oracle else None,"torch_version":torch_version,"owned_bytes":owner.owned_bytes,"actual_imported_semantic_hybrid_sha256":sys.modules[ref.__class__.__module__].__source_sha256__}


def worker(payload):
    with frozen_sources(payload.get("source_identity")) as identity:
        result = _worker(payload)
        result["source_identity"] = identity
        return result


def _worker(payload):
    if payload["mode"] == "oracle":
        return {"gated_delta":functional(oracle=True),"gated_delta_checkpoint_geometry":functional(oracle=True,geometry=(16,32,128,128),steps=4),"weighted_exp":weighted_exp_oracle()}
    mode = payload["mode"]
    if mode not in {"copy_reference","owned"}:
        raise ValueError("invalid worker mode")
    tokens = payload["tokens"]
    if type(tokens) is not int or not 1 <= tokens <= 4096:
        raise ValueError("tokens outside bounded research horizon")
    owner = OwnedGatedDelta(16,32,128,128) if mode == "owned" else None
    ref = owner.geometry if owner is not None else reference_class()(16,32,128,128)
    x = inputs(16,32,128,128)
    state = ref.initial_state() if owner is None else None
    begin_cpu,begin = time.process_time(),time.perf_counter()
    for _ in range(tokens):
        if mode == "copy_reference":
            output,state = ref.evaluate(*x,state,mode="decode")
        else:
            output = owner.step(*x)
    elapsed,cpu = time.perf_counter()-begin,time.process_time()-begin_cpu
    final_state = owner.state if owner else state
    if not np.all(np.isfinite(output)) or not np.all(np.isfinite(final_state)):
        if owner:
            owner.active = False
            owner.state.fill(0)
        raise ValueError("nonfinite timing output/state")
    return {"mode":mode,"tokens":tokens,"repeat":payload["repeat"],"wall_seconds":elapsed,"cpu_seconds":cpu,
        "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=="darwin" else 1024),
        "retained_state_bytes":owner.state.nbytes if owner else state.nbytes,"candidate_owned_bytes":owner.owned_bytes if owner else None,
        "finite_output":True,"finite_state":True,"output_commitment":commitment(output),"state_commitment":commitment(final_state),
        "actual_imported_semantic_hybrid_sha256":sys.modules[ref.__class__.__module__].__source_sha256__,"numpy_version":np.__version__}


def verify_pairs(rows):
    pairs = {}
    for row in rows:
        key = (row["tokens"],row["repeat"])
        if row["mode"] in pairs.setdefault(key,{}):
            raise ValueError("duplicate cohort row")
        pairs[key][row["mode"]] = row
    for pair in pairs.values():
        if set(pair) != {"copy_reference","owned"}:
            raise ValueError("missing strongest copy_reference comparator")
        for field in ("output_commitment","state_commitment","actual_imported_semantic_hybrid_sha256","source_identity"):
            if pair["owned"][field] != pair["copy_reference"][field]:
                raise NumericMismatch(f"numeric/source failure: {field} differs from copy_reference")
    return {"pairs":len(pairs),"exact_output_and_state_bits":True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit",action="store_true")
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--measure",action="store_true")
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("preserve previous evidence")
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    if args.audit:
        result = audit(fetch=True)
    elif args.measure:
        identity = source_identity()
        rows = []
        try:
            with frozen_sources(identity):
                import frontier_probe_support
                helper = Path(frontier_probe_support.__file__).resolve()
                if helper != (ROOT/"scripts/frontier_probe_support.py").resolve() or digest(helper) != identity[str(helper)]:
                    raise ValueError("actual imported measurement helper mismatch")
                fresh_workers = frontier_probe_support.fresh_workers
                with fresh_workers(__file__,memory_bytes=768<<20) as (run,guard,host):
                    oracle = run({"mode":"oracle","source_identity":identity})
                    for tokens in (64,256,1024):
                        for repeat in range(3):
                            for mode in ("copy_reference","owned"):
                                verify_identity(identity)
                                rows.append(run({"mode":mode,"tokens":tokens,"repeat":repeat,"source_identity":identity}))
                                verify_identity(identity)
                            verify_pairs(rows)
                    agreement = verify_pairs(rows)
                    result = {"scope":"local-clear projected one-layer recurrence; no model weights or privacy protocol", "audit":audit(),"oracle":oracle,"measurements":rows,"build":run.build,
                        "source_identity":identity,"source_identity_unchanged":True,"copy_reference_agreement":agreement,
                        "serving_metrics":{key:None for key in ("client_rss","client_cpu","preparation_cpu","preparation_rss","ttft","decode_tps_n_minus_1","all_link_traffic")}}
        except Exception as error:
            if args.output:
                failure = args.output.with_suffix(".failure.json")
                try:
                    after = source_identity()
                    after_error = None
                except Exception as identity_error:
                    after = None
                    after_error = type(identity_error).__name__
                with failure.open("x") as stream:
                    json.dump({"status":"numeric_failure" if isinstance(error,NumericMismatch) else "failed",
                        "error_type":type(error).__name__,"copy_reference_exact_bits":False if isinstance(error,NumericMismatch) else None,
                        "source_identity_before":identity,"source_identity_after":after,"source_identity_after_error":after_error,"measurements":rows},stream,indent=2)
            raise
    else:
        result = functional()
    text = json.dumps(result,indent=2)+"\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError("preserve previous evidence")
        args.output.write_text(text)
    else:
        print(text,end="")


if __name__ == "__main__":
    main()
