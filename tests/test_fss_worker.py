"""ASGI protocol controls with explicit native/source doubles, real encrypted tape.

No installed-worker, model, numerical/security or serving qualification evidence.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, contextmanager
import threading

import httpx
import pytest
import pytest_asyncio

from pllm import Deployment, ExecutionBudget, Model
from pllm.configuration import Experiment
from pllm.profiles import FuseFssSourceCpu
from pllm.runtime import fss_worker as worker, fss_protocol as p

HTTPClient = httpx.AsyncClient
PROFILE = "fusefss.gpt2-complete-ring50-q12.v1"
ORIGINS = {r: f"http://127.0.0.1:{7300+i}" for i, r in enumerate(p.ROLES)}


def configuration():
    return Experiment(name="protocol-only-fss", pipeline=FuseFssSourceCpu(
        Model.path("/generated/protocol-only"), numeric_profile=PROFILE),
        deployment=Deployment.local(root="local://protocol-only"),
        budget=ExecutionBudget(requests=8, max_input_tokens=3, max_new_tokens=3))


def key(sender, recipient):
    return f"{sender}-to-{recipient}-secret-123456789"


def client_key(role):
    return key("client", role)


def connections(role):
    peers = p.HOLDERS if role == "dealer" else tuple(r for r in p.HOLDERS if r != role)
    return {r: (ORIGINS[r], key(role, r)) for r in peers}


def credentials(role):
    return {} if role == "dealer" else {r: key(r, role) for r in p.ROLES if r != role}


def contract(rows=3, outputs=3):
    return p.canonical({"rows": rows, "outputs": outputs, "profile": PROFILE})


class Frozen:
    source_lock_digest = "a"*64
    tokenizer_digest = "b"*64
    snapshot_pins = {"source_lock": "a"*64, "tokenizer": "b"*64,
                     "checkpoint": "c"*64, "config": "d"*64}

    def __init__(self, role):
        self.role = role

    def runtime_contract(self, experiment, rows, outputs):
        return contract(rows, outputs)


class Loaded:
    def __init__(self, role, rows, outputs):
        self.native_program = self
        self.role, self.closed = role, False
        self.contract_bytes = contract(rows, outputs)
        self.manifest_bytes = p.canonical({"schema": "pllm.fss.channel.program.v1",
            "program": "e"*64, "profile": PROFILE, "initial": rows, "selections": outputs})

    def close(self):
        self.closed = True


def callback_metadata(request_id, kind, role, kernel=0):
    meta = {"schema": "pllm.fss.callback.v1", "kind": kind, "role": role,
        "kernel": {"context": {"plan": "e"*64, "session": request_id,
            "operation": "f"*64, "first_tensor_index": 0},
            "plan_digest": "a"*64, "ordinal": kernel}}
    meta.update({"pc": 3, "packet_bytes_bound": 64} if kind == "material"
                else {"frame": 0, "frame_bytes": 7})
    return p.canonical(meta)


@pytest.fixture
def double(monkeypatch):
    class Double:
        making_role = None
        reservations = []
        loaded = []
        natives = []
        native_threads = []
        sealed_roles = set()
        calls = []
        load_fail = False
        load_wait = False
        bad_admission = False
        hold_role = None
        entered = threading.Event()
        exited = threading.Event()
        release = threading.Event()
        load_entered = threading.Event()
        load_release = threading.Event()
        load_cancelled = []
        peer_transform = None
        rpc_log = []

    @contextmanager
    def reserve(size):
        assert size == p.ROLE_MEMORY
        row = {"closed": False, "role": Double.making_role}
        Double.reservations.append(row)
        try:
            yield
        finally:
            assert not row["closed"]
            row["closed"] = True

    def metadata(experiment, *, expected_pins=None):
        if expected_pins is not None and expected_pins != Frozen.snapshot_pins:
            raise ValueError("wrong startup pins")
        return Frozen(Double.making_role)

    def load(experiment, *, expected_pins, source, input_rows, outputs, cancelled):
        assert expected_pins == source.snapshot_pins
        assert any(not r["closed"] for r in Double.reservations)
        Double.load_entered.set()
        if Double.load_wait:
            assert Double.load_release.wait(5)
        Double.load_cancelled.append(cancelled())
        owner = Loaded(source.role, input_rows, outputs)
        Double.loaded.append(owner)
        if Double.load_fail:
            owner.close()  # loader owns its unpublished partial owner
            raise ValueError("injected private partial import detail")
        return owner

    class Native:
        def __init__(self, program, contract_bytes, request, role):
            assert not program.closed and contract_bytes == program.contract_bytes
            assert request.startswith(b"request:")
            self.request_id = request[8:].decode()
            self.role, self.closed = role, False
            self.stop = threading.Event()
            self.callbacks = None
            self.running = False
            Double.natives.append(self)

        def metadata(self):
            return p.canonical({"schema": "pllm.fss.role_admission.v1",
                "role": True if Double.bad_admission else self.role,
                "context": {"plan": "e"*64, "session": self.request_id,
                    "operation": "f"*64, "first_tensor_index": 0},
                "kernels": 2, "primitive_requests_per_party": 2, "frames_per_party": 2,
                "largest_packet_bytes": 64, "cumulative_packet_bytes_per_party_bound": 128,
                "guard_capacity": 10, "combined_reservation_bytes": 64 << 20,
                "peer_frame_bytes_bound": 64, "callback_metadata_bytes_bound": 1024,
                "input_frame_bytes": 272, "output_frame_bytes": 272})

        def run(self, frame, callbacks):
            self.callbacks, self.running = callbacks, True
            Double.native_threads.append(threading.get_ident())
            try:
                if self.role == 2:
                    assert frame is None
                    for kernel in range(2):
                        for holder in (0, 1):
                            callbacks.deliver(callback_metadata(self.request_id, "material", holder, kernel),
                                              f"material-{holder}-{kernel}".encode())
                else:
                    assert frame == f"input-{self.role}".encode()
                    assert Double.sealed_roles == set(p.HOLDERS)
                    Double.entered.set()
                    if Double.hold_role == p.ROLES[self.role]:
                        assert Double.release.wait(5)
                    for kernel in range(2):
                        if self.stop.is_set():
                            raise ValueError("native cancellation")
                        Double.calls.append((self.role, "material"))
                        packet = callbacks.material(callback_metadata(self.request_id, "material", self.role, kernel))
                        assert packet == f"material-{self.role}-{kernel}".encode()
                        other = callbacks.exchange(callback_metadata(self.request_id, "peer", self.role, kernel),
                                                   f"peer{self.role}-{kernel}".encode())
                        assert other == f"peer{1-self.role}-{kernel}".encode()
                callbacks.finish()
                return None if self.role == 2 else f"final-{self.role}".encode()
            finally:
                self.running = False
                if self.role != 2:
                    Double.exited.set()

        def cancel(self):
            self.stop.set()
            if self.callbacks is not None:
                self.callbacks.cancel()

        def close(self):
            assert not self.running
            self.closed = True

    real_seal = worker.MaterialTape.seal
    def seal(tape):
        result = real_seal(tape)
        # Tape identity intentionally has no public role getter. Record at append
        # using canonical metadata, without reading its encrypted backing file.
        Double.sealed_roles.add(tape._test_role)
        return result
    real_append = worker.MaterialTape.append
    def append(tape, meta, packet):
        tape._test_role = p.HOLDERS[p.metadata(meta)["role"]]
        return real_append(tape, meta, packet)

    monkeypatch.setattr(worker, "reserve_incremental_memory", reserve)
    monkeypatch.setattr(worker, "resolve_client_source", metadata)
    monkeypatch.setattr(worker, "load_role_source", load)
    monkeypatch.setattr(worker._native, "FssRoleRequest", Native, raising=False)
    monkeypatch.setattr(worker.MaterialTape, "seal", seal)
    monkeypatch.setattr(worker.MaterialTape, "append", append)
    return Double


class Routing(httpx.AsyncBaseTransport):
    def __init__(self, apps, double):
        self.apps, self.double = apps, double

    async def handle_async_request(self, request):
        role = next(r for r in p.ROLES if request.url.port == 7300+p.ROLES.index(r))
        if self.double.peer_transform is not None:
            request = self.double.peer_transform(request)
        response = await httpx.ASGITransport(app=self.apps[role]).handle_async_request(request)
        sender=next(r for r in p.ROLES if request.headers['authorization']==f'Bearer {key(r,role)}')
        phase='online' if request.url.path.endswith('/peer') else 'setup'
        self.double.rpc_log.append((sender,role,phase,len(request.content),int(response.headers['content-length'])))
        return response


@pytest_asyncio.fixture
async def cluster(double, monkeypatch, request):
    apps = {}
    def outbound(**kwargs):
        assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
        assert kwargs["limits"].max_connections == 1
        return HTTPClient(transport=Routing(apps, double), **kwargs)
    monkeypatch.setattr(worker.httpx, "AsyncClient", outbound)
    async with AsyncExitStack() as stack:
        clients = {}
        exp = configuration()
        for role in p.ROLES:
            double.making_role = role
            apps[role] = worker.create_fss_worker_app(exp, role, client_key(role),
                peer_connections=connections(role), peer_credentials=credentials(role),
                ttl=getattr(request,'param',300))
            await stack.enter_async_context(apps[role].router.lifespan_context(apps[role]))
            clients[role] = await stack.enter_async_context(HTTPClient(
                transport=httpx.ASGITransport(app=apps[role]), base_url=ORIGINS[role],
                headers={"authorization": f"Bearer {client_key(role)}"}))

        class Cluster:
            request_id = "1"*64
            epochs = None

            async def manifests(self):
                values = {r: p.unpack((await clients[r].get('/v1/fss/manifest')).content) for r in p.ROLES}
                self.epochs = {r: m["epoch"] for r, m in values.items()}
                return values

            async def admit(self, roles=p.ROLES, request_id=None, **changes):
                if request_id:
                    self.request_id = request_id
                if self.epochs is None:
                    await self.manifests()
                results = {}
                for r in roles:
                    value = {"request": self.request_id, "epoch": self.epochs[r], "epochs": self.epochs,
                             "rows": 3, "outputs": 3, "contract": contract(), **changes}
                    results[r] = await clients[r].post('/v1/fss/admit', content=p.pack(value, limit=p.ADMISSION_LIMIT))
                return results

            async def post(self, role, action, value=None, *, raw=None, request_id=None, credential=None):
                return await clients[role].post(f'/v1/fss/{request_id or self.request_id}/{action}',
                    content=raw if raw is not None else (b'' if value is None else p.pack(value, limit=p.FRAME_LIMIT)),
                    headers={} if credential is None else {"authorization": f"Bearer {credential}"})

            async def bind(self, roles=p.ROLES):
                for r in roles:
                    response = await self.post(r, 'bind', {"request_frame": b'request:'+self.request_id.encode()})
                    assert response.status_code == 200, response.content

            async def ready(self):
                assert all(r.status_code == 200 for r in (await self.admit()).values())
                await self.bind()

            async def prepare(self):
                response = await asyncio.wait_for(self.post('dealer', 'prepare'), 5)
                assert response.status_code == 200, response.content
                assert p.unpack(response.content) == {"prepared": True}

            async def inputs(self):
                for i,r in enumerate(p.HOLDERS):
                    assert (await self.post(r,'input',raw=f'input-{i}'.encode())).status_code == 200

            async def metrics(self, role):
                return (await clients[role].get('/v1/fss/metrics')).json()

        c = Cluster()
        c.clients, c.apps, c.double = clients, apps, double
        yield c
        double.release.set()
        double.load_release.set()
    assert all(r["closed"] for r in double.reservations)
    assert all(x.closed for x in double.loaded)
    assert all(x.closed for x in double.natives)


@pytest.mark.asyncio
@pytest.mark.parametrize('first',p.HOLDERS)
async def test_offline_barrier_then_direct_holder_execution_and_exact_counters(cluster, first):
    manifests = await cluster.manifests()
    assert all(m['schema']=='pllm.fss.worker.v1' and m['transport']==p.TRANSPORT for m in manifests.values())
    await cluster.ready()
    assert not cluster.double.calls
    await cluster.prepare()  # no input/native holder run needed to acknowledge all material
    assert cluster.double.sealed_roles == set(p.HOLDERS)
    dealer = p.unpack((await cluster.post('dealer','finish')).content)
    assert dealer['frame'] == b'' and dealer['storage'] == {}
    assert (await cluster.metrics('dealer'))['active_sessions']==0
    await cluster.inputs()
    assert (await cluster.post(first,'start')).status_code==200
    await asyncio.sleep(.02)  # first incoming frame may arrive before recipient starts
    other = next(r for r in p.HOLDERS if r!=first)
    assert (await cluster.post(other,'start')).status_code==200
    replies = await asyncio.wait_for(asyncio.gather(*(cluster.post(r,'finish') for r in p.HOLDERS)),5)
    assert all(r.status_code==200 for r in replies)
    values = [p.unpack(r.content) for r in replies]
    from pllm.runtime.fss_transport import _receipt
    for i,value in enumerate(values):
        assert value['frame']==f'final-{i}'.encode()
        assert value['storage']['state']=='finished'
        assert value['storage']['records_written']==value['storage']['records_read']==2
        assert _receipt(replies[i].content, p.HOLDERS[i])[2] == value['storage']
        assert value['peer_bytes']['dealer']==dict.fromkeys(value['peer_bytes']['dealer'],0)
    totals = dict.fromkeys(p.PEER_COUNTERS,0)
    for r,value in zip(p.ROLES,[*values,dealer]):
        expected=p.peer_bytes()
        for sender,recipient,phase,upload,download in cluster.double.rpc_log:
            if sender==r:
                expected[recipient][f'{phase}_upload_bytes']+=upload
                expected[recipient][f'{phase}_download_bytes']+=download
        assert value['peer_bytes']==expected
        for name,count in p.collect_peer_bytes(value['peer_bytes'],r).items():
            totals[name]+=count
    assert totals['peer_link.dealer.worker_a.setup_bytes']>0
    assert totals['peer_link.worker_a.dealer.setup_bytes']>0  # ACKs counted by dealer only
    assert totals['peer_link.worker_a.worker_b.online_bytes']>0
    assert all(totals[f'peer_link.dealer.{r}.online_bytes']==0 for r in p.HOLDERS)
    assert all(t!=threading.get_ident() for t in cluster.double.native_threads)


@pytest.mark.asyncio
@pytest.mark.parametrize('role,action', [('worker_a','input'),('worker_a','start'),('dealer','input')])
async def test_early_or_dealer_input_burns_attempt(cluster,role,action):
    await cluster.ready()
    response=await cluster.post(role,action,raw=b'private' if action=='input' else b'')
    assert response.status_code==400
    assert (await cluster.metrics(role))['active_sessions']==0
    assert not cluster.double.calls


@pytest.mark.asyncio
async def test_auth_stale_session_and_no_dealer_incoming_channels(cluster):
    await cluster.ready()
    assert (await cluster.post('worker_a','material',{},credential=client_key('worker_a'))).status_code==401
    assert (await cluster.post('dealer','peer',{},credential=key('worker_a','dealer'))).status_code==401
    assert (await cluster.post('worker_a','peer',{},request_id='9'*64,
        credential=key('worker_b','worker_a'))).status_code==409
    assert (await cluster.metrics('worker_a'))['active_sessions']==1
    # Authenticated but wrong direction cannot act as dealer.
    assert (await cluster.post('worker_a','material',{},credential=key('worker_b','worker_a'))).status_code==400
    assert (await cluster.metrics('worker_a'))['active_sessions']==0


@pytest.mark.asyncio
async def test_partial_source_failure_burns_session_and_rolls_back_only_local_role(cluster):
    assert (await cluster.admit(roles=('dealer',)))['dealer'].status_code==200
    cluster.double.load_fail=True
    assert (await cluster.admit(roles=('worker_a',)))['worker_a'].status_code==400
    assert (await cluster.metrics('dealer'))['active_sessions']==1
    assert (await cluster.metrics('worker_a'))['active_sessions']==0
    cluster.double.load_fail=False
    assert (await cluster.admit(roles=('worker_a',)))['worker_a'].status_code==400
    assert (await cluster.admit(roles=('worker_a',),request_id='2'*64))['worker_a'].status_code==200
    assert (await cluster.post('dealer','cancel',request_id='1'*64)).status_code==200


@pytest.mark.asyncio
@pytest.mark.parametrize('change',['epoch','contract','bool_rows','same_epochs'])
async def test_admission_checks_before_payload_load(cluster,change):
    await cluster.manifests()
    changes={'epoch':'f'*64} if change=='epoch' else {'contract':b'wrong'} if change=='contract' else (
        {'rows':True} if change=='bool_rows' else {'epochs':dict.fromkeys(p.ROLES,cluster.epochs['worker_a'])})
    response=(await cluster.admit(roles=('worker_a',),**changes))['worker_a']
    assert response.status_code==400
    assert not cluster.double.loaded
    assert (await cluster.metrics('worker_a'))['active_sessions']==0


@pytest.mark.asyncio
async def test_native_admission_before_tape_and_second_bind_burns(cluster):
    await cluster.admit()
    cluster.double.bad_admission=True
    reply=await cluster.post('worker_a','bind',{'request_frame':b'request:'+cluster.request_id.encode()})
    assert reply.status_code==400
    assert cluster.double.natives[-1].closed
    cluster.double.bad_admission=False
    await cluster.bind(roles=('dealer',))
    assert (await cluster.post('dealer','bind',{'request_frame':b'request:'+cluster.request_id.encode()})).status_code==400
    assert (await cluster.metrics('dealer'))['active_sessions']==0


@pytest.mark.asyncio
@pytest.mark.parametrize('mutation',['sequence','role','epoch','metadata','seal_early'])
async def test_authenticated_material_errors_burn_complete_owner(cluster,mutation):
    await cluster.ready()
    message={'sender':'dealer','sender_epoch':cluster.epochs['dealer'],
        'recipient_epoch':cluster.epochs['worker_a'],'sequence':0,
        'metadata':callback_metadata(cluster.request_id,'material',0),'packet':b'packet'}
    action='material'
    if mutation=='sequence': message['sequence']=1
    elif mutation=='role': message['metadata']=callback_metadata(cluster.request_id,'material',1)
    elif mutation=='epoch': message['sender_epoch']='f'*64
    elif mutation=='metadata': message['metadata']=message['metadata']+b' '
    else:
        action='seal'; del message['metadata']; del message['packet']
    assert (await cluster.post('worker_a',action,message,credential=key('dealer','worker_a'))).status_code==400
    assert (await cluster.metrics('worker_a'))['active_sessions']==0
    assert next(x for x in cluster.double.loaded if x.role=='worker_a').closed


@pytest.mark.asyncio
async def test_material_replay_and_no_material_after_seal(cluster):
    await cluster.ready()
    message={'sender':'dealer','sender_epoch':cluster.epochs['dealer'],
        'recipient_epoch':cluster.epochs['worker_a'],'sequence':0,
        'metadata':callback_metadata(cluster.request_id,'material',0),'packet':b'packet'}
    assert (await cluster.post('worker_a','material',message,credential=key('dealer','worker_a'))).status_code==200
    assert (await cluster.post('worker_a','material',message,credential=key('dealer','worker_a'))).status_code==400


@pytest.mark.asyncio
async def test_material_after_seal_is_forbidden(cluster):
    await cluster.ready(); await cluster.prepare()
    assert (await cluster.post('worker_a','material',{},credential=key('dealer','worker_a'))).status_code==400
    assert (await cluster.metrics('worker_a'))['active_sessions']==0


@pytest.mark.asyncio
async def test_tape_creation_failure_closes_already_admitted_native_owner(cluster,monkeypatch):
    await cluster.admit()
    real=worker.MaterialTape
    class BrokenTape:
        price=staticmethod(real.price)
        def __init__(self,*args,**kwargs): raise ValueError('disk unavailable')
    monkeypatch.setattr(worker,'MaterialTape',BrokenTape)
    response=await cluster.post('worker_a','bind',{'request_frame':b'request:'+cluster.request_id.encode()})
    assert response.status_code==400
    assert cluster.double.natives[-1].closed
    assert not cluster.double.calls
    assert (await cluster.metrics('worker_a'))['active_sessions']==0


@pytest.mark.asyncio
async def test_cancel_waits_for_native_thread_before_refund(cluster):
    await cluster.ready(); await cluster.prepare(); await cluster.inputs()
    cluster.double.hold_role='worker_a'
    assert (await cluster.post('worker_a','start')).status_code==200
    assert await asyncio.to_thread(cluster.double.entered.wait,2)
    loaded=next(x for x in cluster.double.loaded if x.role=='worker_a')
    before=[r for r in cluster.double.reservations if not r['closed']]
    cancel=asyncio.create_task(cluster.post('worker_a','cancel'))
    await asyncio.sleep(.03)
    assert not cancel.done() and not loaded.closed
    assert all(not r['closed'] for r in before)
    cluster.double.release.set()
    assert (await asyncio.wait_for(cancel,3)).status_code==200
    assert loaded.closed and cluster.double.exited.is_set()
    assert (await cluster.metrics('worker_a'))['active_sessions']==0
    assert (await cluster.post('worker_a','cancel')).status_code==200


@pytest.mark.asyncio
async def test_two_cancel_callers_join_same_cleanup_without_self_join(cluster):
    await cluster.ready(); await cluster.prepare(); await cluster.inputs()
    cluster.double.hold_role='worker_a'
    assert (await cluster.post('worker_a','start')).status_code==200
    assert await asyncio.to_thread(cluster.double.entered.wait,2)
    first=asyncio.create_task(cluster.post('worker_a','cancel'))
    await asyncio.sleep(.01)
    second=asyncio.create_task(cluster.post('worker_a','cancel'))
    await asyncio.sleep(.01)
    assert not first.done() and not second.done()
    cluster.double.release.set()
    assert all(r.status_code==200 for r in await asyncio.wait_for(asyncio.gather(first,second),3))


@pytest.mark.asyncio
async def test_cancelled_source_import_drains_late_returned_owner(cluster):
    cluster.double.load_wait=True
    task=asyncio.create_task(cluster.admit(roles=('worker_a',)))
    assert await asyncio.to_thread(cluster.double.load_entered.wait,2)
    held=next(r for r in reversed(cluster.double.reservations) if not r['closed'])
    task.cancel(); await asyncio.sleep(.02); task.cancel()
    assert not task.done() and not held['closed']
    cluster.double.load_release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task,3)
    assert held['closed'] and cluster.double.loaded[-1].closed
    assert cluster.double.load_cancelled[-1] is True
    assert (await cluster.metrics('worker_a'))['active_sessions']==0


@pytest.mark.asyncio
async def test_seen_ledger_saved_request_bound_and_no_retry(cluster):
    for n in range(1,9):
        response=(await cluster.admit(roles=('worker_a',),request_id=f'{n:064x}',contract=b'wrong'))['worker_a']
        assert response.status_code==400
    assert (await cluster.admit(roles=('worker_a',),request_id='f'*64))['worker_a'].status_code==409
    assert (await cluster.metrics('worker_a'))['seen_requests']==8


@pytest.mark.asyncio
async def test_peer_out_of_order_burns_but_ready_mailbox_accepts_one(cluster):
    await cluster.ready(); await cluster.prepare(); await cluster.inputs()
    message={'sender':'worker_b','epochs':cluster.epochs,'sequence':1,
        'metadata':callback_metadata(cluster.request_id,'peer',1),'frame':b'peer1-0'}
    assert (await cluster.post('worker_a','peer',message,credential=key('worker_b','worker_a'))).status_code==400
    assert (await cluster.metrics('worker_a'))['active_sessions']==0


@pytest.mark.asyncio
async def test_peer_matching_rejects_other_plan_before_native_reply(cluster):
    await cluster.ready(); await cluster.prepare(); await cluster.inputs()
    def mutate(request):
        if request.url.path.endswith('/peer'):
            value=p.unpack(request.content,limit=p.FRAME_LIMIT)
            meta=p.metadata(value['metadata'])
            meta['kernel']['plan_digest']='d'*64
            value['metadata']=p.canonical(meta)
            return httpx.Request(request.method,request.url,headers=request.headers,content=p.pack(value,limit=p.FRAME_LIMIT))
        return request
    cluster.double.peer_transform=mutate
    for role in p.HOLDERS:
        assert (await cluster.post(role,'start')).status_code==200
    for _ in range(200):
        a=await cluster.metrics('worker_a'); b=await cluster.metrics('worker_b')
        if a['active_sessions']==b['active_sessions']==0: break
        await asyncio.sleep(.005)
    assert a['active_sessions']==b['active_sessions']==0
    assert all(n.closed for n in cluster.double.natives if n.role in (0,1))


@pytest.mark.asyncio
async def test_mailbox_backpressure_reads_no_second_body_and_cancel_drains(cluster,monkeypatch):
    await cluster.ready(); await cluster.prepare(); await cluster.inputs()
    def message(sequence):
        return {'sender':'worker_b','epochs':cluster.epochs,'sequence':sequence,
            'metadata':callback_metadata(cluster.request_id,'peer',1,sequence),'frame':f'peer1-{sequence}'.encode()}
    assert (await cluster.post('worker_a','peer',message(0),credential=key('worker_b','worker_a'))).status_code==200
    reads=[]
    old_body=worker._body
    async def body(request,limit):
        if request.path_params.get('action')=='peer': reads.append(True)
        return await old_body(request,limit)
    monkeypatch.setattr(worker,'_body',body)
    waiting=asyncio.create_task(cluster.post('worker_a','peer',message(1),credential=key('worker_b','worker_a')))
    await asyncio.sleep(.02)
    assert not waiting.done() and not reads
    assert (await asyncio.wait_for(cluster.post('worker_a','cancel'),2)).status_code==200
    await asyncio.gather(waiting,return_exceptions=True)
    assert (await cluster.metrics('worker_a'))['active_sessions']==0


@pytest.mark.asyncio
@pytest.mark.parametrize('cluster',[.5],indirect=True)
async def test_ttl_retires_bound_source_and_tape_without_timer_self_join(cluster):
    await cluster.ready()
    await asyncio.sleep(.65)
    assert all(x.closed for x in cluster.double.loaded)
    assert all(x.closed for x in cluster.double.natives)
    assert (await cluster.metrics('dealer'))['active_sessions']==0


@pytest.mark.asyncio
async def test_shutdown_joins_blocked_native_before_owner_refund(cluster):
    await cluster.ready(); await cluster.prepare(); await cluster.inputs()
    cluster.double.hold_role='worker_a'
    assert (await cluster.post('worker_a','start')).status_code==200
    assert await asyncio.to_thread(cluster.double.entered.wait,2)
    loaded=next(x for x in cluster.double.loaded if x.role=='worker_a')
    # The lifespan has no startup mutation; nested exit directly exercises the
    # same real shutdown path while the fixture retains final cleanup ownership.
    context=cluster.apps['worker_a'].router.lifespan_context(cluster.apps['worker_a'])
    await context.__aenter__()
    shutdown=asyncio.create_task(context.__aexit__(None,None,None))
    await asyncio.sleep(.02)
    assert not shutdown.done() and not loaded.closed
    cluster.double.release.set()
    await asyncio.wait_for(shutdown,3)
    assert loaded.closed


@pytest.mark.asyncio
async def test_single_active_and_bounded_body_before_import(cluster):
    await cluster.ready()
    assert (await cluster.clients['dealer'].post('/v1/fss/admit',content=b'bad')).status_code==409
    response=await cluster.clients['worker_a'].post(f'/v1/fss/{cluster.request_id}/input',content=b'bad',
        headers={'content-length':str(273)})
    assert response.status_code==400
    assert (await cluster.metrics('worker_a'))['active_sessions']==0


@pytest.mark.asyncio
async def test_finish_response_retains_reservation_through_blocked_send(cluster):
    await cluster.ready(); await cluster.prepare()
    held=[r for r in cluster.double.reservations if not r['closed']]
    sending, release=asyncio.Event(),asyncio.Event()
    messages=[]
    scope={'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','method':'POST',
        'scheme':'http','path':f'/v1/fss/{cluster.request_id}/finish','raw_path':b'/',
        'query_string':b'','root_path':'','server':('127.0.0.1',7302),'client':('127.0.0.1',1),
        'headers':[(b'authorization',f'Bearer {client_key("dealer")}'.encode()),(b'content-length',b'0')]}
    async def receive(): return {'type':'http.request','body':b'','more_body':False}
    async def send(message):
        messages.append(message)
        if message['type']=='http.response.body':
            sending.set(); await release.wait()
    task=asyncio.create_task(cluster.apps['dealer'](scope,receive,send))
    try:
        await asyncio.wait_for(sending.wait(),2)
        assert all(not r['closed'] for r in held)
        assert next(x for x in cluster.double.loaded if x.role=='dealer').closed
    finally:
        release.set(); await asyncio.wait_for(task,2)
    assert sum(r['closed'] for r in held)==1
    assert next(m for m in messages if m['type']=='http.response.start')['status']==200


@pytest.mark.parametrize('change',['duplicate_key','dns','remote_http','extra_incoming','extra_outgoing'])
def test_channels_fail_closed_before_source_and_reservation(double,change):
    c,credentials_map=connections('dealer'),credentials('dealer')
    if change=='duplicate_key': c['worker_a']=(ORIGINS['worker_a'],client_key('dealer'))
    elif change=='dns': c['worker_a']=('https://example.com',key('dealer','worker_a'))
    elif change=='remote_http': c['worker_a']=('http://192.0.2.1',key('dealer','worker_a'))
    elif change=='extra_incoming': credentials_map['worker_a']=key('worker_a','dealer')
    else: c['dealer']=('http://127.0.0.1:7309',key('dealer','dealer'))
    with pytest.raises(ValueError):
        worker.create_fss_worker_app(configuration(),'dealer',client_key('dealer'),
            peer_connections=c,peer_credentials=credentials_map)
    assert not double.reservations
