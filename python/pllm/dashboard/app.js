const $ = (q, root=document) => root.querySelector(q);
const $$ = (q, root=document) => [...root.querySelectorAll(q)];
let snapshot = null;
let clockTimer = null;
let protocolCursor = 0;
let sequenceSpans = [];

function bytes(n=0) {
  if (n < 1024) return `${Math.round(n)} B`;
  if (n < 1048576) return `${(n/1024).toFixed(1)} KB`;
  if (n < 1073741824) return `${(n/1048576).toFixed(1)} MB`;
  return `${(n/1073741824).toFixed(2)} GB`;
}
function cpuTime(seconds=0) {
  if (seconds < 10) return `${seconds.toFixed(2)}s`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${(seconds % 60).toFixed(0)}s`;
}
function actorTraffic(traffic, actor) {
  let sent = 0, received = 0;
  Object.entries(traffic).forEach(([edge, amount]) => {
    const [source, destination] = edge.split("->");
    if (source === actor) sent += Number(amount || 0);
    if (destination === actor) received += Number(amount || 0);
  });
  return {sent, received};
}
function spark(path, history) {
  if (!history?.length) { path.setAttribute("d", "M0 43 L300 43"); return; }
  const values = history.map(x => x.cpu || 0), max = Math.max(5, ...values);
  const points = values.map((v,i) => `${i ? "L" : "M"}${i*300/Math.max(1,values.length-1)} ${42-v*39/max}`);
  path.setAttribute("d", points.join(" "));
}
function phaseText(phase) {
  return ({starting:"STARTING REAL SERVICES",preparing:"PREPARING OFFLINE INVENTORY",ready:"READY / INVENTORY SEALED",online:"PRIVATE INFERENCE ONLINE",refilling:"REFILLING OFFLINE INVENTORY",error:"ATTENTION REQUIRED"})[phase] || phase;
}
function shortStage(stage="linear") {
  return (stage || "linear").replace(/^model\./, "").replace(/layers\.(\d+)\./, "L$1 / ").replaceAll("_proj", "");
}
function flowRow(source, destination, label, amount, kind) {
  const row = document.createElement("div");
  row.className = `sequence-message ${source}-${destination}`;
  const route = document.createElement("span");
  route.className = `sequence-route ${kind} ${source === "inference" || destination === "client" ? "reverse" : ""}`;
  const copy = document.createElement("b");
  copy.textContent = `${label} · ${bytes(amount)}`;
  route.append(copy);
  row.append(route);
  return row;
}
function renderSequence(incoming, cursor) {
  const container = $("#sequence-events");
  if (cursor < protocolCursor) {
    protocolCursor = 0;
    sequenceSpans = [];
    container.replaceChildren();
    return;
  }
  sequenceSpans.push(...incoming);
  protocolCursor = cursor;
  const operations = sequenceSpans
    .filter(span => span.name === "pllm.prepared_linear" && span.flows)
    .sort((left, right) => (left.start - right.start) || (left.sequence - right.sequence));
  const additions = incoming
    .filter(span => span.name === "pllm.prepared_linear" && span.flows)
    .sort((left, right) => left.sequence - right.sequence);
  if (!operations.length) {
    const empty = document.createElement("p");
    empty.className = "sequence-empty";
    empty.textContent = "Preparing inventory operations.";
    container.replaceChildren(empty);
    $("#sequence-count").textContent = "WAITING FOR PREPARATION";
    return;
  }
  const cards = additions.map(span => {
    const card = document.createElement("article");
    card.className = `sequence-operation ${span.phase === "offline" ? "offline" : "online"}`;
    const head = document.createElement("div");
    head.className = "sequence-operation-head";
    const stage = document.createElement("strong");
    stage.textContent = shortStage(span.stage);
    const phase = document.createElement("span");
    phase.className = "sequence-phase";
    phase.textContent = span.phase === "offline" ? "OFFLINE PREP" : "ONLINE";
    const duration = document.createElement("span");
    duration.textContent = `${span.duration_ms.toFixed(1)} ms`;
    head.append(stage, phase, duration);
    const flows = span.phase === "offline"
      ? [
          flowRow("client", "preparation", "seed batch", span.flows.client_preparation || 0, "offline"),
          flowRow("preparation", "inference", "W·r−s", span.flows.preparation_inference || 0, "offline"),
          flowRow("preparation", "client", "durable ACK", span.flows.preparation_client || 0, "offline")
        ]
      : [
          flowRow("client", "inference", "ticket + x-r", span.flows.client_inference || 0, "masked"),
          flowRow("inference", "client", "Wx-s", span.flows.inference_client || 0, "masked")
        ];
    card.append(head, ...flows);
    return card;
  });
  if (!container.querySelector(".sequence-operation")) container.replaceChildren(...cards);
  else container.append(...cards);
  const prepared = operations.filter(span => span.phase === "offline").length;
  $("#sequence-count").textContent = `ALL ${operations.length.toLocaleString()} OPS · ${prepared.toLocaleString()} PREP`;
  cards.at(-1)?.scrollIntoView({block:"nearest"});
}
function render(data) {
  snapshot = data;
  const run = data.run, otel = data.otel;
  document.body.classList.toggle("running", run.phase === "online");
  $("#topology").classList.toggle("running", run.phase === "online");
  $(".status").classList.toggle("live", ["ready","online","refilling"].includes(run.phase));
  $("#status-copy").textContent = phaseText(run.phase);
  $("#run").disabled = run.phase !== "ready" || !run.processes?.inference?.running;
  $("#run").textContent = run.phase === "online" ? "RUNNING…" : "RUN PRIVATE CHAT  →";
  $("#answer").textContent = run.text || (run.phase === "online" ? "Computing first token…" : "Waiting for a live run.");
  $("#model-label").textContent = `MODEL / ${run.model_id || "unknown"}${run.tiny ? " / RANDOM-WEIGHT TRANSPORT TEST" : ""}`;
  $("#error").textContent = run.error || "";
  $("#tps").textContent = Number(run.tps || 0).toFixed(2);
  $("#ttft").textContent = run.ttft_seconds == null ? "--" : `${run.ttft_seconds.toFixed(2)}s`;
  const traffic = otel.traffic || {}, onlineTraffic = run.online_traffic || {};
  $("#client-prep-bytes").textContent = bytes((traffic["client->preparation"]||0)+(traffic["preparation->client"]||0));
  $("#relay-bytes").textContent = bytes(traffic["preparation->inference"]||run.privacy?.correction_push_bytes||0);
  $("#inference-bytes").textContent = bytes((onlineTraffic["client->inference"]||0)+(onlineTraffic["inference->client"]||0));
  const inventory = run.inventory || {};
  $("#inventory-capacity").textContent = Number(inventory.capacity || 0).toLocaleString();
  $("#inventory-available").textContent = Number(inventory.available || 0).toLocaleString();
  $("#inventory-reserved").textContent = Number(inventory.reserved || 0).toLocaleString();
  $("#inventory-burned").textContent = Number(inventory.burned || 0).toLocaleString();
  $("#inventory-consumed").textContent = Number(inventory.consumed || 0).toLocaleString();
  $("#prep-online-cpu").textContent = `${Number(run.preparation_online_operations || 0).toLocaleString()} GEMMs`;
  $$(".service").forEach(card => {
    const service = otel.services[card.dataset.service] || {};
    const actor = card.dataset.service.replace("pllm-", "");
    const totals = actorTraffic(traffic, actor);
    const work = actor === "client"
      ? run.privacy?.online_steps
      : actor === "preparation"
        ? run.privacy?.preparation_rows
        : run.privacy?.inference_stage_calls;
    $('[data-metric="cpu-time"]',card).textContent = cpuTime(Number(service.cpu_seconds || 0));
    $('[data-metric="sent"]',card).textContent = bytes(totals.sent);
    $('[data-metric="received"]',card).textContent = bytes(totals.received);
    $('[data-metric="work"]',card).textContent = Number(work || 0).toLocaleString();
    const cpuMetric = $('[data-metric="cpu"]', card);
    if (cpuMetric) cpuMetric.textContent = `${Number(service.cpu_percent||0).toFixed(1)}%`;
    $('[data-metric="memory"]',card).textContent = bytes(service.memory_bytes||0);
    spark($("[data-spark]",card), service.history);
  });
  renderSequence(otel.protocol_spans || [], Number(otel.protocol_cursor || 0));
}
async function poll() {
  try { const response = await fetch(`/api/snapshot?protocol_after=${protocolCursor}`, {cache:"no-store"}); render(await response.json()); }
  catch (error) {
    console.error("dashboard poll failed", error);
    $("#status-copy").textContent = "COLLECTOR OFFLINE / KEEP CLI RUNNING";
  }
}
async function run() {
  const prompt = $("#prompt").value.trim(), max = Number($("#max-tokens").value);
  if (!prompt) return;
  $("#run").disabled = true;
  const response = await fetch("/api/run", {method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({prompt,max_output_tokens:max})});
  if (!response.ok) $("#error").textContent = (await response.json()).detail || "Run failed";
  await poll();
}
$("#run").addEventListener("click", run);
$("#prompt").addEventListener("keydown", e => { if ((e.metaKey||e.ctrlKey) && e.key === "Enter") run(); });
clockTimer = setInterval(() => {
  if (!snapshot?.run?.started_at) return;
  const end = snapshot.run.finished_at || Date.now()/1000, elapsed = Math.max(0,end-snapshot.run.started_at);
  $("#run-clock").textContent = `${String(Math.floor(elapsed/60)).padStart(2,"0")}:${(elapsed%60).toFixed(1).padStart(4,"0")}`;
},100);
poll(); setInterval(poll,500);
