const $ = (query, root = document) => root.querySelector(query);
const $$ = (query, root = document) => [...root.querySelectorAll(query)];
const SVG_NS = "http://www.w3.org/2000/svg";
const CHART_COLORS = ["#0f766e", "#b45309", "#2563eb", "#be123c", "#6d28d9", "#3f6212"];
const MAX_SEQUENCE_OPERATIONS = 500;
const MAX_RETAINED_SPANS = 65536;
const MAX_PROTOCOL_RUNS = 12;

let snapshot = null;
let clockTimer = null;
let protocolCursor = 0;
let sequenceSpans = [];
let activeProtocolKey = "session";
let protocolGeneration = 0;
let sequenceDisplayKey = "";
let pollInFlight = false;
let historyInFlight = false;
let selectionRequest = 0;
let runSerial = 0;
let lastPhase = null;

const protocolRuns = new Map([[activeProtocolKey, makeProtocolState(0)]]);
const dashboard = {
  runs: [],
  selectedId: null,
  selectedRecord: null,
  selectedRecordComplete: false,
  historyState: "loading",
  historyPageSize: 200,
  activeRunId: null,
};

const METRICS = {
  duration: {
    label: "Total time",
    matrixLabel: "Median total time",
    value: recordDuration,
    format: formatDuration,
    axis: compactDuration,
  },
  ttft: {
    label: "Time to first token",
    matrixLabel: "Median TTFT",
    value: recordTtft,
    format: formatDuration,
    axis: compactDuration,
  },
  tps: {
    label: "Decode throughput",
    matrixLabel: "Median decode rate",
    value: recordTps,
    format: value => `${value.toFixed(2)} tok/s`,
    axis: value => value.toFixed(value < 10 ? 1 : 0),
  },
  online_bytes: {
    label: "Online masked I/O",
    matrixLabel: "Median online I/O",
    value: recordOnlineBytes,
    format: bytes,
    axis: compactBytes,
  },
  cpu: {
    label: "Aggregate CPU time",
    matrixLabel: "Median aggregate CPU",
    value: recordCpuSeconds,
    format: cpuTime,
    axis: compactDuration,
  },
};

function object(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function valueAt(source, path) {
  if (!source || typeof source !== "object") return undefined;
  if (typeof path === "string" && Object.prototype.hasOwnProperty.call(source, path)) {
    return source[path];
  }
  const parts = Array.isArray(path) ? path : String(path).split(".");
  let current = source;
  for (const part of parts) {
    if (!current || typeof current !== "object" || !Object.prototype.hasOwnProperty.call(current, part)) {
      return undefined;
    }
    current = current[part];
  }
  return current;
}

function pick(sources, paths, fallback = null) {
  for (const source of sources) {
    for (const path of paths) {
      const value = valueAt(source, path);
      if (value !== undefined && value !== null) return value;
    }
  }
  return fallback;
}

function numberOrNull(value) {
  if (value === null || value === undefined || value === "" || typeof value === "boolean") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function firstNumber(values) {
  for (const value of values) {
    const parsed = numberOrNull(value);
    if (parsed !== null) return parsed;
  }
  return null;
}

function integer(value, fallback = 0) {
  const parsed = numberOrNull(value);
  return parsed === null ? fallback : Math.round(parsed);
}

function bytes(value = 0) {
  const amount = Math.max(0, Number(value) || 0);
  if (amount < 1024) return `${Math.round(amount)} B`;
  if (amount < 1048576) return `${(amount / 1024).toFixed(1)} KB`;
  if (amount < 1073741824) return `${(amount / 1048576).toFixed(1)} MB`;
  return `${(amount / 1073741824).toFixed(2)} GB`;
}

function compactBytes(value) {
  const amount = Math.max(0, Number(value) || 0);
  if (amount < 1024) return `${Math.round(amount)} B`;
  if (amount < 1048576) return `${(amount / 1024).toFixed(amount < 10240 ? 1 : 0)}K`;
  if (amount < 1073741824) return `${(amount / 1048576).toFixed(amount < 10485760 ? 1 : 0)}M`;
  return `${(amount / 1073741824).toFixed(1)}G`;
}

function cpuTime(value = 0) {
  const seconds = Math.max(0, Number(value) || 0);
  if (seconds < 10) return `${seconds.toFixed(2)}s`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${(seconds % 60).toFixed(0)}s`;
}

function formatDuration(value) {
  const seconds = numberOrNull(value);
  if (seconds === null) return "--";
  if (seconds < 0.001) return `${(seconds * 1000000).toFixed(0)} us`;
  if (seconds < 1) return `${(seconds * 1000).toFixed(seconds < 0.1 ? 1 : 0)} ms`;
  return cpuTime(seconds);
}

function compactDuration(value) {
  const seconds = Math.max(0, Number(value) || 0);
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  return `${(seconds / 60).toFixed(1)}m`;
}

function epochSeconds(value) {
  if (typeof value === "number" && Number.isFinite(value)) return value > 1e12 ? value / 1000 : value;
  if (typeof value === "string" && value) {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) return numeric > 1e12 ? numeric / 1000 : numeric;
    const parsed = Date.parse(value);
    if (Number.isFinite(parsed)) return parsed / 1000;
  }
  return null;
}

function formatTimestamp(value) {
  const seconds = epochSeconds(value);
  if (seconds === null) return "--";
  const date = new Date(seconds * 1000);
  if (!Number.isFinite(date.getTime())) return "--";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

function titleCase(value) {
  const text = String(value || "unknown").replaceAll("_", " ").replaceAll("-", " ");
  return text.replace(/\b\w/g, letter => letter.toUpperCase());
}

function recordParts(record) {
  const root = object(record);
  const run = object(root.run);
  const metrics = object(pick([run, root], ["metrics", "measurements"], {}));
  const timing = object(pick([run, root, metrics], ["timing", "times"], {}));
  const timestamps = object(pick([run, root, timing], ["timestamps"], {}));
  const durations = object(pick([run, root, metrics, timing], ["durations"], {}));
  const usage = object(pick([run, root, metrics], ["usage", "tokens"], {}));
  const otel = object(pick([root, run], ["otel", "telemetry"], {}));
  return {root, run, metrics, timing, timestamps, durations, usage, otel};
}

function recordId(record) {
  const {root, run} = recordParts(record);
  const id = pick([root, run], ["run_id", "id", "uuid"], "");
  return id === "" ? "" : String(id);
}

function recordModel(record) {
  const {root, run, metrics} = recordParts(record);
  return String(pick([run, root, metrics], ["model_id", "model", "model_name"], "unknown"));
}

function recordFingerprint(record) {
  const {root, run, metrics} = recordParts(record);
  const value = pick([run, root, metrics], ["model_fingerprint", "fingerprint"], null);
  return value === null ? null : String(value);
}

function recordCohort(record) {
  const {root, run} = recordParts(record);
  const cold = pick([root, run], ["cold"], null);
  const mode = typeof cold === "boolean" ? (cold ? "cold" : "warm") : "mode unknown";
  const fingerprint = recordFingerprint(record);
  return {
    key: `${recordModel(record)}\u0000${fingerprint || "unknown"}\u0000${mode}`,
    label: `${recordModel(record)} / ${mode}${fingerprint ? ` / ${fingerprint.slice(0, 8)}` : ""}`,
  };
}

function recordPhase(record) {
  const {root, run} = recordParts(record);
  return String(pick([run, root], ["phase", "status", "state"], "unknown"));
}

function recordActualContext(record) {
  const {root, run, metrics, usage} = recordParts(record);
  return firstNumber([
    run.actual_context_tokens,
    root.actual_context_tokens,
    metrics.actual_context_tokens,
    run.context_tokens,
    root.context_tokens,
    metrics.context_tokens,
    usage.input_tokens,
    usage.prompt_tokens,
    usage.input,
  ]);
}

function recordOutputTokens(record) {
  const {root, run, metrics, usage} = recordParts(record);
  return firstNumber([
    run.output_tokens,
    root.output_tokens,
    metrics.output_tokens,
    run.generated_tokens,
    root.generated_tokens,
    usage.output_tokens,
    usage.completion_tokens,
    usage.output,
    run.tokens,
  ]);
}

function recordStarted(record) {
  const {root, run, timing, timestamps} = recordParts(record);
  const direct = pick([run, root, timing, timestamps], ["full_started_at", "started_at", "start_time", "created_at", "timestamp"]);
  if (direct !== null) return direct;
  const nanoseconds = numberOrNull(timestamps.started_at_ns ?? run.started_at_ns ?? root.started_at_ns);
  return nanoseconds === null ? null : nanoseconds / 1e9;
}

function recordOnlineStarted(record) {
  const {root, run, timing, timestamps} = recordParts(record);
  const direct = pick([run, root, timing, timestamps], ["online_started_at", "started_at", "start_time"]);
  if (direct !== null) return direct;
  const nanoseconds = numberOrNull(timestamps.online_started_at_ns);
  return nanoseconds === null ? recordStarted(record) : nanoseconds / 1e9;
}

function recordFirstToken(record) {
  const {root, run, timing, timestamps} = recordParts(record);
  const direct = pick([run, root, timing, timestamps], ["first_token_at", "first_token_time"]);
  if (direct !== null) return direct;
  const nanoseconds = numberOrNull(timestamps.first_token_at_ns ?? run.first_token_at_ns ?? root.first_token_at_ns);
  return nanoseconds === null ? null : nanoseconds / 1e9;
}

function recordLastToken(record) {
  const {root, run, timing, timestamps} = recordParts(record);
  const direct = pick([run, root, timing, timestamps], ["last_token_at", "last_token_time"]);
  if (direct !== null) return direct;
  const nanoseconds = numberOrNull(timestamps.last_token_at_ns ?? run.last_token_at_ns ?? root.last_token_at_ns);
  return nanoseconds === null ? null : nanoseconds / 1e9;
}

function recordFinished(record) {
  const {root, run, timing, timestamps} = recordParts(record);
  const direct = pick([run, root, timing, timestamps], ["finished_at", "finish_time", "completed_at", "ended_at"]);
  if (direct !== null) return direct;
  const nanoseconds = numberOrNull(timestamps.finished_at_ns ?? run.finished_at_ns ?? root.finished_at_ns);
  return nanoseconds === null ? null : nanoseconds / 1e9;
}

function recordDuration(record) {
  const {root, run, metrics, timing, durations} = recordParts(record);
  const direct = numberOrNull(pick(
    [run, root, metrics, timing, durations],
    ["duration_seconds", "total_seconds", "full_seconds", "elapsed_seconds", "wall_seconds", "duration"],
  ));
  if (direct !== null) return direct;
  const start = epochSeconds(recordStarted(record));
  const finish = epochSeconds(recordFinished(record));
  return start !== null && finish !== null ? Math.max(0, finish - start) : null;
}

function recordTtft(record) {
  const {root, run, metrics, timing, durations} = recordParts(record);
  const direct = numberOrNull(pick(
    [run, root, metrics, timing, durations],
    ["ttft_seconds", "time_to_first_token_seconds", "first_token_seconds", "ttft"],
  ));
  if (direct !== null) return direct;
  const start = epochSeconds(recordOnlineStarted(record));
  const first = epochSeconds(recordFirstToken(record));
  return start !== null && first !== null ? Math.max(0, first - start) : null;
}

function recordTps(record) {
  const {root, run, metrics, timing, durations} = recordParts(record);
  return numberOrNull(pick(
    [run, root, metrics, timing, durations],
    ["tps", "tokens_per_second", "decode_tokens_per_second", "throughput"],
  ));
}

function firstMap(candidates) {
  let empty = null;
  for (const candidate of candidates) {
    if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) continue;
    if (Object.keys(candidate).length) return candidate;
    empty = empty || candidate;
  }
  return empty || {};
}

function trafficMaps(record) {
  const {root, run, metrics, otel} = recordParts(record);
  const rootTraffic = object(root.traffic);
  const runTraffic = object(run.traffic);
  const io = object(pick([run, root, metrics], ["io", "network"], {}));
  const totalSource = firstMap([
      otel.traffic,
      root.total_traffic,
      run.total_traffic,
      rootTraffic.total,
      runTraffic.total,
      root.traffic,
      run.traffic,
      io.total,
    ]);
  const hadTotalTraffic = Object.keys(totalSource).length > 0;
  const total = {...totalSource};
  const onlineSource = firstMap([
      run.online_traffic,
      root.online_traffic,
      metrics.online_traffic,
      rootTraffic.online,
      runTraffic.online,
      io.online,
      otel.online_traffic,
    ]);
  const hadOnlineTraffic = Object.keys(onlineSource).length > 0;
  const online = {...onlineSource};
  const privacy = object(pick([run, root, metrics], ["privacy", "privacy_audit", "audit"], {}));
  const setMissing = (map, edge, value) => {
    if (edgeAmount(map, edge) !== null) return;
    const measured = numberOrNull(value);
    if (measured !== null) map[edge] = measured;
  };
  setMissing(total, "client->preparation", privacy.preparation_upload_bytes);
  setMissing(total, "preparation->client", privacy.preparation_download_bytes);
  setMissing(total, "preparation->inference", privacy.correction_push_bytes);
  setMissing(online, "client->inference", privacy.masked_online_upload_bytes ?? privacy.inference_upload_bytes);
  setMissing(online, "inference->client", privacy.masked_online_download_bytes ?? privacy.inference_download_bytes);
  if (!hadTotalTraffic && Object.keys(online).length) Object.assign(total, online);
  return {total, online, hadOnlineTraffic};
}

function edgeAmount(traffic, edge) {
  const map = object(traffic);
  const [source, destination] = edge.split("->");
  return numberOrNull(pick(
    [map, object(map[source])],
    [edge, `${source}_to_${destination}`, `${source}_${destination}`, destination],
  ));
}

function privacyFor(record) {
  const {root, run, metrics} = recordParts(record);
  return object(pick([run, root, metrics], ["privacy", "privacy_audit", "audit"], {}));
}

function onlineEdgeAmount(record, edge) {
  const online = trafficMaps(record).online;
  const measured = edgeAmount(online, edge);
  if (measured !== null) return measured;
  const privacy = privacyFor(record);
  const fallback = edge === "client->inference"
    ? pick([privacy], ["masked_online_upload_bytes", "inference_upload_bytes"])
    : pick([privacy], ["masked_online_download_bytes", "inference_download_bytes"]);
  return numberOrNull(fallback);
}

function recordOnlineBytes(record) {
  const direct = numberOrNull(pick(
    [recordParts(record).run, recordParts(record).root, recordParts(record).metrics],
    ["online_bytes", "online_io_bytes", "masked_online_bytes"],
  ));
  if (direct !== null) return direct;
  const traffic = trafficMaps(record);
  if (!traffic.hadOnlineTraffic) {
    const privacy = privacyFor(record);
    const maskedUpload = numberOrNull(privacy.masked_online_upload_bytes);
    const maskedDownload = numberOrNull(privacy.masked_online_download_bytes);
    if (maskedUpload !== null || maskedDownload !== null) {
      return (maskedUpload || 0) + (maskedDownload || 0);
    }
  }
  const upload = onlineEdgeAmount(record, "client->inference");
  const download = onlineEdgeAmount(record, "inference->client");
  return upload === null && download === null ? null : (upload || 0) + (download || 0);
}

function servicesFor(record) {
  const {root, run, metrics, otel} = recordParts(record);
  return firstMap([
    otel.services,
    root.resources,
    run.resources,
    metrics.resources,
    root.services,
    root.processes,
    run.processes,
  ]);
}

function serviceFor(record, serviceName) {
  const services = servicesFor(record);
  const actor = serviceName.replace("pllm-", "");
  return object(pick([services], [serviceName, actor], {}));
}

function recordCpuSeconds(record) {
  const direct = numberOrNull(pick(
    [recordParts(record).run, recordParts(record).root, recordParts(record).metrics],
    ["cpu_seconds", "total_cpu_seconds", "aggregate_cpu_seconds"],
  ));
  if (direct !== null) return direct;
  let found = false;
  const total = ["pllm-client", "pllm-preparation", "pllm-inference"].reduce((sum, name) => {
    const value = numberOrNull(pick([serviceFor(record, name)], ["cpu_seconds", "cpu_time_seconds", "cpu_time"]));
    if (value !== null) found = true;
    return sum + (value || 0);
  }, 0);
  return found ? total : null;
}

function actorTraffic(traffic, actor) {
  let sent = 0;
  let received = 0;
  Object.entries(object(traffic)).forEach(([edge, amount]) => {
    const [source, destination] = edge.split("->");
    if (source === actor) sent += Number(amount || 0);
    if (destination === actor) received += Number(amount || 0);
  });
  return {sent, received};
}

function phaseText(phase) {
  return ({
    starting: "STARTING REAL SERVICES",
    preparing: "PREPARING OFFLINE INVENTORY",
    ready: "READY / INVENTORY SEALED",
    online: "PRIVATE INFERENCE ONLINE",
    refilling: "REFILLING OFFLINE INVENTORY",
    completed: "RUN COMPLETE",
    error: "ATTENTION REQUIRED",
    failed: "RUN FAILED",
  })[phase] || String(phase || "UNKNOWN").toUpperCase();
}

function shortStage(stage = "linear") {
  return String(stage || "linear")
    .replace(/^model\./, "")
    .replace(/layers\.(\d+)\./, "L$1 / ")
    .replaceAll("_proj", "");
}

function spark(path, history) {
  const samples = Array.isArray(history) ? history : [];
  if (!samples.length) {
    path.setAttribute("d", "M0 39 L300 39");
    return;
  }
  const values = samples.map(item => Math.max(0, Number(object(item).cpu || 0)));
  const maximum = Math.max(5, ...values);
  const points = values.map((value, index) => {
    const x = index * 300 / Math.max(1, values.length - 1);
    const y = 38 - value * 35 / maximum;
    return `${index ? "L" : "M"}${x} ${y}`;
  });
  path.setAttribute("d", points.join(" "));
}

function makeProtocolState(cursor) {
  return {cursor: Math.max(0, Number(cursor) || 0), spans: [], seen: new Set(), discarded: 0};
}

function currentProtocolState() {
  if (!protocolRuns.has(activeProtocolKey)) protocolRuns.set(activeProtocolKey, makeProtocolState(protocolCursor));
  return protocolRuns.get(activeProtocolKey);
}

function syncProtocolGlobals() {
  const state = currentProtocolState();
  protocolCursor = state.cursor;
  sequenceSpans = state.spans;
}

function startProtocolRun(key) {
  const baseline = currentProtocolState().cursor;
  protocolGeneration += 1;
  activeProtocolKey = key;
  protocolRuns.set(key, makeProtocolState(baseline));
  syncProtocolGlobals();
  trimProtocolRuns();
  if (dashboard.selectedId === null) paintSequence([], `live:${key}`, "Waiting for run operations.");
}

function trimProtocolRuns() {
  while (protocolRuns.size > MAX_PROTOCOL_RUNS) {
    const oldest = protocolRuns.keys().next().value;
    if (oldest === activeProtocolKey) {
      const retained = protocolRuns.get(oldest);
      protocolRuns.delete(oldest);
      protocolRuns.set(oldest, retained);
    } else {
      protocolRuns.delete(oldest);
    }
  }
}

function renameProtocolRun(nextKey) {
  if (!nextKey || nextKey === activeProtocolKey) return;
  if (protocolRuns.has(nextKey)) {
    protocolRuns.delete(activeProtocolKey);
    activeProtocolKey = nextKey;
    syncProtocolGlobals();
    trimProtocolRuns();
    return;
  }
  const state = currentProtocolState();
  protocolRuns.delete(activeProtocolKey);
  activeProtocolKey = nextKey;
  protocolRuns.set(nextKey, state);
  syncProtocolGlobals();
  trimProtocolRuns();
}

function protocolSpanKey(span) {
  const item = object(span);
  if (item.sequence !== undefined && item.sequence !== null) return `sequence:${item.sequence}`;
  const flows = Object.entries(object(item.flows)).sort(([left], [right]) => left.localeCompare(right));
  return JSON.stringify([item.name, item.stage, item.phase, item.start, item.time, item.duration_ms, flows]);
}

function protocolOperations(spans) {
  return (Array.isArray(spans) ? spans : [])
    .filter(span => object(span).name === "pllm.prepared_linear" && object(span).flows)
    .sort((left, right) => {
      const leftSequence = numberOrNull(object(left).sequence);
      const rightSequence = numberOrNull(object(right).sequence);
      if (leftSequence !== null && rightSequence !== null) return leftSequence - rightSequence;
      return (Number(object(left).start) || 0) - (Number(object(right).start) || 0);
    });
}

function flowRow(source, destination, label, amount, kind) {
  const row = document.createElement("div");
  row.className = `sequence-message ${source}-${destination}`;
  const route = document.createElement("span");
  route.className = `sequence-route ${kind} ${source === "inference" || destination === "client" ? "reverse" : ""}`;
  const copy = document.createElement("b");
  copy.textContent = `${label} / ${bytes(amount)}`;
  route.append(copy);
  row.append(route);
  return row;
}

function operationCard(rawSpan) {
  const span = object(rawSpan);
  const offline = span.phase === "offline";
  const card = document.createElement("article");
  card.className = `sequence-operation ${offline ? "offline" : "online"}`;

  const head = document.createElement("div");
  head.className = "sequence-operation-head";
  const stage = document.createElement("strong");
  stage.textContent = shortStage(span.stage);
  const phase = document.createElement("span");
  phase.className = "sequence-phase";
  phase.textContent = offline ? "OFFLINE PREP" : "ONLINE";
  const duration = document.createElement("span");
  duration.textContent = `${Math.max(0, Number(span.duration_ms) || 0).toFixed(1)} ms`;
  head.append(stage, phase, duration);

  const flows = object(span.flows);
  const rows = offline
    ? [
        flowRow("client", "preparation", "seed batch", flows.client_preparation || 0, "offline"),
        flowRow("preparation", "inference", "W*r-s", flows.preparation_inference || 0, "offline"),
        flowRow("preparation", "client", "durable ACK", flows.preparation_client || 0, "offline"),
      ]
    : [
        flowRow("client", "inference", "ticket + x-r", flows.client_inference || 0, "masked"),
        flowRow("inference", "client", "Wx-s", flows.inference_client || 0, "masked"),
      ];
  card.append(head, ...rows);
  return card;
}

function updateSequenceCount(spans, discarded = 0) {
  const operations = protocolOperations(spans);
  if (!operations.length) {
    $("#sequence-count").textContent = "OTEL / WAITING";
    return;
  }
  const prepared = operations.filter(span => object(span).phase === "offline").length;
  const online = operations.length - prepared;
  const total = operations.length + discarded;
  const retained = discarded ? ` / ${operations.length.toLocaleString()} RETAINED` : "";
  $("#sequence-count").textContent = `${total.toLocaleString()} CAPTURED${retained} / ${prepared.toLocaleString()} PREP / ${online.toLocaleString()} ONLINE`;
}

function paintSequence(spans, displayKey, emptyCopy = "No protocol operations recorded for this run.", discarded = 0) {
  const container = $("#sequence-events");
  const operations = protocolOperations(spans);
  const fragment = document.createDocumentFragment();
  if (!operations.length) {
    const empty = document.createElement("p");
    empty.className = "sequence-empty";
    empty.textContent = emptyCopy;
    fragment.append(empty);
  } else {
    operations.slice(-MAX_SEQUENCE_OPERATIONS).forEach(span => fragment.append(operationCard(span)));
  }
  container.replaceChildren(fragment);
  sequenceDisplayKey = displayKey;
  updateSequenceCount(spans, discarded);
}

function appendSequence(additions, allSpans, displayKey, discarded = 0) {
  if (sequenceDisplayKey !== displayKey) {
    paintSequence(allSpans, displayKey);
    return;
  }
  const operations = protocolOperations(additions);
  if (operations.length) {
    const container = $("#sequence-events");
    $(".sequence-empty", container)?.remove();
    const fragment = document.createDocumentFragment();
    operations.slice(-MAX_SEQUENCE_OPERATIONS).forEach(span => fragment.append(operationCard(span)));
    container.append(fragment);
    while (container.childElementCount > MAX_SEQUENCE_OPERATIONS) container.firstElementChild.remove();
  }
  updateSequenceCount(allSpans, discarded);
}

function renderSequence(incoming, cursor, runKey = activeProtocolKey, generation = protocolGeneration, truncated = false, oldest = null) {
  if (generation !== protocolGeneration) return;
  if (!protocolRuns.has(runKey)) protocolRuns.set(runKey, makeProtocolState(protocolCursor));
  const state = protocolRuns.get(runKey);
  const nextCursor = Math.max(0, Number(cursor) || 0);
  let reset = nextCursor < state.cursor;
  if (reset) {
    state.cursor = 0;
    state.spans = [];
    state.seen = new Set();
    state.discarded = 0;
  }
  if (truncated) {
    const oldestSequence = numberOrNull(oldest);
    const missing = oldestSequence === null
      ? Math.max(0, nextCursor - state.cursor - incoming.length)
      : Math.max(0, oldestSequence - state.cursor - 1);
    state.discarded = Math.max(state.discarded, missing);
  }

  const additions = [];
  for (const rawSpan of Array.isArray(incoming) ? incoming : []) {
    const span = object(rawSpan);
    const key = protocolSpanKey(span);
    if (state.seen.has(key)) continue;
    state.seen.add(key);
    state.spans.push(span);
    additions.push(span);
  }
  if (state.spans.length > MAX_RETAINED_SPANS) {
    const removed = state.spans.splice(0, state.spans.length - MAX_RETAINED_SPANS);
    removed.forEach(span => state.seen.delete(protocolSpanKey(span)));
    state.discarded += removed.length;
  }
  state.cursor = nextCursor;
  if (runKey === activeProtocolKey) syncProtocolGlobals();

  if (dashboard.selectedId !== null || runKey !== activeProtocolKey) return;
  const displayKey = `live:${runKey}`;
  if (reset) paintSequence(state.spans, displayKey, undefined, state.discarded);
  else appendSequence(additions, state.spans, displayKey, state.discarded);
}

function protocolSpansFor(record) {
  const {root, run, otel} = recordParts(record);
  const candidates = [
    otel.protocol_spans,
    root.protocol_spans,
    run.protocol_spans,
    root.network_sequence,
    valueAt(root, "network.sequence"),
  ];
  return candidates.find(Array.isArray) || [];
}

function showLiveSequence() {
  const state = currentProtocolState();
  paintSequence(state.spans, `live:${activeProtocolKey}`, "Run a chat to capture protocol operations.", state.discarded);
}

function setText(selector, value) {
  const element = $(selector);
  const next = String(value);
  if (element && element.textContent !== next) element.textContent = next;
}

function renderTopology(record, historical = false) {
  const {run} = recordParts(record);
  const {total, online} = trafficMaps(record);
  const privacy = privacyFor(record);

  const clientPreparation = (edgeAmount(total, "client->preparation") || 0) + (edgeAmount(total, "preparation->client") || 0);
  const relay = edgeAmount(total, "preparation->inference")
    ?? numberOrNull(privacy.correction_push_bytes)
    ?? 0;
  const onlineAmount = (edgeAmount(online, "client->inference") || 0) + (edgeAmount(online, "inference->client") || 0);
  setText("#client-prep-bytes", bytes(clientPreparation));
  setText("#relay-bytes", bytes(relay));
  setText("#inference-bytes", bytes(onlineAmount));
  setText("#tps", (recordTps(record) || 0).toFixed(2));
  setText("#ttft", recordTtft(record) === null ? "--" : formatDuration(recordTtft(record)));
  setText("#prep-online-cpu", `${integer(pick([run], ["preparation_online_operations"], 0)).toLocaleString()} GEMMs`);
  setText("#topology-source", historical ? "SELECTED PERSISTED RUN" : "CURRENT LIVE RUN");

  $$(".service").forEach(card => {
    const serviceName = card.dataset.service;
    const actor = serviceName.replace("pllm-", "");
    const service = serviceFor(record, serviceName);
    const totals = actorTraffic(total, actor);
    const work = actor === "client"
      ? privacy.online_steps
      : actor === "preparation"
        ? privacy.preparation_rows
        : privacy.inference_stage_calls;
    const cpuSeconds = numberOrNull(pick([service], ["cpu_seconds", "cpu_time_seconds", "cpu_time"])) || 0;
    const memory = numberOrNull(pick([service], ["memory_bytes", "rss_bytes", "rss_peak_bytes", "peak_memory_bytes"])) || 0;
    const cpu = numberOrNull(pick([service], ["cpu_percent", "cpu_utilization_percent", "cpu"])) || 0;
    const status = pick([service], ["status", "state"], historical ? "recorded" : "waiting");
    $('[data-metric="cpu-time"]', card).textContent = cpuTime(cpuSeconds);
    $('[data-metric="sent"]', card).textContent = bytes(totals.sent);
    $('[data-metric="received"]', card).textContent = bytes(totals.received);
    $('[data-metric="work"]', card).textContent = integer(work).toLocaleString();
    const cpuMetric = $('[data-metric="cpu"]', card);
    if (cpuMetric) cpuMetric.textContent = `${cpu.toFixed(1)}%`;
    $('[data-metric="memory"]', card).textContent = bytes(memory);
    $('[data-metric="status"]', card).textContent = String(status).toUpperCase();
    spark($("[data-spark]", card), service.history);
  });
}

function renderKpis(record, historical) {
  const context = recordActualContext(record);
  const output = recordOutputTokens(record);
  const duration = recordDuration(record);
  const ttft = recordTtft(record);
  const tps = recordTps(record);
  const onlineBytes = recordOnlineBytes(record);
  const id = recordId(record);
  const started = recordStarted(record);

  setText("#kpi-state", titleCase(recordPhase(record)));
  setText("#kpi-model", recordModel(record));
  setText("#kpi-context", context === null ? "--" : integer(context).toLocaleString());
  setText("#kpi-output", output === null ? "--" : integer(output).toLocaleString());
  setText("#kpi-duration", formatDuration(duration));
  setText("#kpi-ttft", formatDuration(ttft));
  setText("#kpi-tps", tps === null ? "--" : tps.toFixed(2));
  setText("#kpi-io", onlineBytes === null ? "--" : bytes(onlineBytes));
  setText("#selection-title", historical ? "Selected persisted run" : "Current live run");
  setText(
    "#selection-meta",
    historical
      ? `${id ? `Run ${id} / ` : ""}${formatTimestamp(started)}`
      : "Live snapshot / archive excludes prompt and output",
  );
}

function appendMeasure(list, label, value) {
  const term = document.createElement("dt");
  const description = document.createElement("dd");
  term.textContent = label;
  description.textContent = value;
  list.append(term, description);
}

function renderMeasureList(selector, entries) {
  const list = $(selector);
  const fragment = document.createDocumentFragment();
  entries.forEach(([label, value]) => appendMeasure(fragment, label, value));
  list.replaceChildren(fragment);
}

function renderTiming(record) {
  const first = epochSeconds(recordFirstToken(record));
  const last = epochSeconds(recordLastToken(record));
  const decode = first !== null && last !== null ? Math.max(0, last - first) : null;
  const context = recordActualContext(record);
  const output = recordOutputTokens(record);
  const {root, run, durations, usage} = recordParts(record);
  const preparation = numberOrNull(pick([durations, run, root], ["preparation_seconds"]));
  const transition = numberOrNull(pick([durations, run, root], ["transition_seconds"]));
  const online = numberOrNull(pick([durations, run, root], ["online_seconds"]));
  const generation = numberOrNull(pick([durations, run, root], ["generation_seconds"]));
  const cold = pick([root, run], ["cold"], null);
  const mode = typeof cold === "boolean" ? (cold ? "Cold" : "Warm") : "--";
  const authoritative = pick([usage], ["authoritative"], null);
  renderMeasureList("#timing-detail", [
    ["Run mode", mode],
    ["Started", formatTimestamp(recordStarted(record))],
    ["Finished", formatTimestamp(recordFinished(record))],
    ["Full request", formatDuration(recordDuration(record))],
    ["Preparation", formatDuration(preparation)],
    ["Phase transition", formatDuration(transition)],
    ["Online", formatDuration(online)],
    ["TTFT", formatDuration(recordTtft(record))],
    ["Generation", formatDuration(generation ?? decode)],
    ["Actual context", context === null ? "--" : `${integer(context).toLocaleString()} tokens`],
    ["Output", output === null ? "--" : `${integer(output).toLocaleString()} tokens`],
    ["Decode rate", recordTps(record) === null ? "--" : `${recordTps(record).toFixed(2)} tok/s`],
    ["Token counts", authoritative === null ? "--" : authoritative ? "response usage" : "reservation / stream fallback"],
  ]);
}

function renderIo(record) {
  const total = trafficMaps(record).total;
  const clientInference = onlineEdgeAmount(record, "client->inference");
  const inferenceClient = onlineEdgeAmount(record, "inference->client");
  const privacy = privacyFor(record);
  const maskedUpload = numberOrNull(privacy.masked_online_upload_bytes);
  const maskedDownload = numberOrNull(privacy.masked_online_download_bytes);
  renderMeasureList("#io-detail", [
    ["Client -> preparation", edgeAmount(total, "client->preparation") === null ? "--" : bytes(edgeAmount(total, "client->preparation"))],
    ["Preparation -> client", edgeAmount(total, "preparation->client") === null ? "--" : bytes(edgeAmount(total, "preparation->client"))],
    ["Preparation -> inference", edgeAmount(total, "preparation->inference") === null ? "--" : bytes(edgeAmount(total, "preparation->inference"))],
    ["Client -> inference / online", clientInference === null ? "--" : bytes(clientInference)],
    ["Inference -> client / online", inferenceClient === null ? "--" : bytes(inferenceClient)],
    ["Masked payload upload", maskedUpload === null ? "--" : bytes(maskedUpload)],
    ["Masked payload download", maskedDownload === null ? "--" : bytes(maskedDownload)],
    ["Online masked total", recordOnlineBytes(record) === null ? "--" : bytes(recordOnlineBytes(record))],
  ]);
}

function inventoryFor(record) {
  const {root, run, metrics} = recordParts(record);
  return object(pick([run, root, metrics], ["inventory", "prepared_inventory"], {}));
}

function renderInventory(record) {
  const inventory = inventoryFor(record);
  const inventoryValue = key => {
    const value = numberOrNull(inventory[key]);
    return value === null ? "--" : integer(value).toLocaleString();
  };
  const persistedShape = ["required", "generated", "reused"].some(key => numberOrNull(inventory[key]) !== null);
  const entries = persistedShape
    ? [
        ["Required", inventoryValue("required")],
        ["Generated", inventoryValue("generated")],
        ["Reused", inventoryValue("reused")],
        ["Consumed", inventoryValue("consumed")],
        ["Burned", inventoryValue("burned")],
      ]
    : [
        ["State", titleCase(inventory.status || "unknown")],
        ["Capacity", inventoryValue("capacity")],
        ["Available", inventoryValue("available")],
        ["Reserved", inventoryValue("reserved")],
        ["Consumed", inventoryValue("consumed")],
        ["Burned", inventoryValue("burned")],
      ];
  renderMeasureList("#inventory-detail", entries);
}

function renderResources(record) {
  const entries = [];
  for (const [serviceName, label] of [
    ["pllm-client", "Client"],
    ["pllm-preparation", "Preparation"],
    ["pllm-inference", "Inference"],
  ]) {
    const service = serviceFor(record, serviceName);
    const cpu = numberOrNull(pick([service], ["cpu_seconds", "cpu_time_seconds", "cpu_time"]));
    const memory = numberOrNull(pick([service], ["memory_bytes", "rss_bytes", "rss_peak_bytes", "peak_memory_bytes"]));
    const threads = numberOrNull(pick([service], ["threads", "thread_count"]));
    entries.push([`${label} CPU`, cpu === null ? "--" : cpuTime(cpu)]);
    entries.push([`${label} RAM`, memory === null ? "--" : bytes(memory)]);
    if (threads !== null) entries.push([`${label} threads`, integer(threads).toLocaleString()]);
  }
  renderMeasureList("#resource-detail", entries);
}

function humanLabel(value) {
  return titleCase(String(value).replaceAll(".", " "));
}

function measuredValue(value, key = "") {
  if (typeof value === "boolean") return value ? "true" : "false";
  const numeric = numberOrNull(value);
  if (numeric !== null) return key.toLowerCase().includes("byte") ? bytes(numeric) : numeric.toLocaleString();
  return value === null || value === undefined ? "--" : String(value);
}

function addInvariant(container, item) {
  const row = document.createElement("article");
  row.className = `invariant-row ${item.passed === true ? "pass" : item.passed === false ? "fail" : "measured"}`;
  const copy = document.createElement("div");
  const label = document.createElement("strong");
  const expectation = document.createElement("small");
  const value = document.createElement("b");
  const state = document.createElement("span");
  label.textContent = item.label;
  expectation.textContent = item.expectation || "observed";
  value.textContent = item.value;
  state.textContent = item.passed === true ? "PASS" : item.passed === false ? "FAIL" : "MEASURED";
  copy.append(label, expectation);
  row.append(copy, value, state);
  container.append(row);
}

function explicitInvariants(record) {
  const {root, run, metrics} = recordParts(record);
  return pick(
    [run, root, metrics],
    ["measured_privacy_invariants", "privacy_invariants", "invariants"],
    null,
  );
}

function normalizeExplicitInvariant(raw, fallbackName) {
  if (typeof raw === "boolean") {
    return {label: humanLabel(fallbackName), value: raw ? "true" : "false", expectation: "must hold", passed: raw};
  }
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return {label: humanLabel(fallbackName), value: measuredValue(raw, fallbackName), expectation: "observed", passed: null};
  }
  const item = object(raw);
  const label = String(pick([item], ["label", "name", "invariant"], humanLabel(fallbackName)));
  const observed = pick([item], ["observed", "measured", "actual", "value"], null);
  const expected = pick([item], ["expected", "expectation", "requirement"], null);
  const explicitPass = pick([item], ["passed", "pass", "holds", "ok"], null);
  let passed = typeof explicitPass === "boolean" ? explicitPass : null;
  if (passed === null && expected !== null && observed !== null) passed = String(observed) === String(expected);
  return {
    label,
    value: measuredValue(observed, fallbackName),
    expectation: expected === null ? "observed" : `expected ${measuredValue(expected, fallbackName)}`,
    passed,
  };
}

function renderPrivacy(record) {
  const container = $("#privacy-detail");
  const fragment = document.createDocumentFragment();
  const explicit = explicitInvariants(record);
  let count = 0;

  if (Array.isArray(explicit)) {
    explicit.forEach((raw, index) => {
      addInvariant(fragment, normalizeExplicitInvariant(raw, `invariant ${index + 1}`));
      count += 1;
    });
  } else if (explicit && typeof explicit === "object") {
    Object.entries(explicit).forEach(([name, raw]) => {
      addInvariant(fragment, normalizeExplicitInvariant(raw, name));
      count += 1;
    });
  }

  if (!count) {
    const privacy = privacyFor(record);
    const run = recordParts(record).run;
    const zeroChecks = [
      ["Plaintext prompt bytes sent", "plaintext_prompt_bytes_sent"],
      ["Plaintext token IDs sent", "plaintext_token_ids_sent"],
      ["Public context bytes sent", "public_context_bytes"],
      ["Preparation requests during online", "preparation_requests_during_online"],
    ];
    zeroChecks.forEach(([label, key]) => {
      const measured = numberOrNull(privacy[key]);
      if (measured === null) return;
      addInvariant(fragment, {
        label,
        value: key.includes("bytes") ? bytes(measured) : measured.toLocaleString(),
        expectation: "expected 0",
        passed: measured === 0,
      });
      count += 1;
    });
    const onlinePreparation = numberOrNull(run.preparation_online_operations);
    if (onlinePreparation !== null) {
      addInvariant(fragment, {
        label: "Preparation matrix work during online phase",
        value: onlinePreparation.toLocaleString(),
        expectation: "expected 0",
        passed: onlinePreparation === 0,
      });
      count += 1;
    }
    for (const [label, key] of [
      ["Masked online upload", "masked_online_upload_bytes"],
      ["Masked online download", "masked_online_download_bytes"],
      ["Prepared correlations", "correlation_count"],
      ["Online protocol steps", "online_steps"],
    ]) {
      const measured = numberOrNull(privacy[key]);
      if (measured === null) continue;
      addInvariant(fragment, {
        label,
        value: key.includes("bytes") ? bytes(measured) : measured.toLocaleString(),
        expectation: "observed",
        passed: null,
      });
      count += 1;
    }
  }

  if (!count) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No measured privacy counters in this record.";
    fragment.append(empty);
  }
  container.replaceChildren(fragment);
}

function renderDetail(record) {
  renderTiming(record);
  renderIo(record);
  renderInventory(record);
  renderResources(record);
  renderPrivacy(record);
}

function renderSelection(record, historical, renderNetwork = true) {
  renderKpis(record, historical);
  renderDetail(record);
  renderTopology(record, historical);
  if (!renderNetwork) return;
  if (historical) {
    const id = recordId(record) || "unknown";
    paintSequence(protocolSpansFor(record), `history:${id}`);
  } else {
    showLiveSequence();
  }
}

function renderCurrent(record, renderNetwork = true) {
  const id = recordId(object(record).run);
  const persisted = id ? dashboard.runs.find(item => recordId(item) === id) : null;
  const measurements = persisted || record;
  renderKpis(measurements, false);
  renderDetail(measurements);
  renderTopology(record, false);
  if (renderNetwork) showLiveSequence();
}

function median(values) {
  const sorted = values.filter(Number.isFinite).sort((left, right) => left - right);
  if (!sorted.length) return null;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function selectedMetric() {
  return METRICS[$("#metric-select").value] || METRICS.duration;
}

function comparableRuns() {
  return dashboard.runs.filter(record => (
    recordPhase(record) === "completed"
    && recordActualContext(record) !== null
  ));
}

function renderMatrix() {
  const head = $("#matrix-head");
  const body = $("#matrix-body");
  const table = $("#matrix-table");
  const empty = $("#matrix-empty");
  const records = comparableRuns();
  const metric = selectedMetric();
  setText("#matrix-metric-label", metric.matrixLabel);
  head.replaceChildren();
  body.replaceChildren();

  if (!records.length) {
    table.hidden = true;
    empty.hidden = false;
    return;
  }

  const contexts = [...new Set(records.map(recordActualContext))].sort((left, right) => left - right);
  const cohorts = [...new Map(records.map(record => {
    const cohort = recordCohort(record);
    return [cohort.key, cohort];
  })).values()].sort((left, right) => left.label.localeCompare(right.label));
  const headerRow = document.createElement("tr");
  const corner = document.createElement("th");
  corner.scope = "col";
  corner.textContent = "Cohort / tokens";
  headerRow.append(corner);
  contexts.forEach(context => {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = integer(context).toLocaleString();
    headerRow.append(cell);
  });
  head.append(headerRow);

  const fragment = document.createDocumentFragment();
  cohorts.forEach(cohort => {
    const row = document.createElement("tr");
    const label = document.createElement("th");
    label.scope = "row";
    label.textContent = cohort.label;
    row.append(label);
    contexts.forEach(context => {
      const matching = records.filter(record => recordCohort(record).key === cohort.key && recordActualContext(record) === context);
      const values = matching.map(metric.value).filter(Number.isFinite);
      const aggregate = median(values);
      const cell = document.createElement("td");
      const value = document.createElement("b");
      const count = document.createElement("small");
      value.textContent = aggregate === null ? "--" : metric.format(aggregate);
      count.textContent = values.length ? `n=${values.length}` : "";
      cell.title = values.length ? `${values.length} comparable run${values.length === 1 ? "" : "s"}` : "No comparable measurement";
      cell.append(value, count);
      row.append(cell);
    });
    fragment.append(row);
  });
  body.append(fragment);
  table.hidden = false;
  empty.hidden = true;
}

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
  return element;
}

function renderChart() {
  const svg = $("#comparison-chart");
  const empty = $("#chart-empty");
  const legend = $("#chart-legend");
  const metric = selectedMetric();
  const points = comparableRuns()
    .map(record => ({
      record,
      cohort: recordCohort(record),
      context: recordActualContext(record),
      value: metric.value(record),
    }))
    .filter(point => point.context !== null && Number.isFinite(point.value));
  svg.replaceChildren();
  legend.replaceChildren();

  if (!points.length) {
    svg.hidden = true;
    empty.hidden = false;
    return;
  }

  svg.hidden = false;
  empty.hidden = true;
  const title = svgElement("title");
  title.textContent = `${metric.label} by actual context tokens`;
  svg.append(title);

  const width = 720;
  const height = 300;
  const margin = {top: 18, right: 18, bottom: 46, left: 72};
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const xValues = points.map(point => point.context);
  const yValues = points.map(point => point.value);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMax = Math.max(...yValues, 0.000001) * 1.08;
  const xPosition = value => margin.left + (xMin === xMax ? 0.5 : (value - xMin) / (xMax - xMin)) * plotWidth;
  const yPosition = value => margin.top + plotHeight - value / yMax * plotHeight;

  for (let index = 0; index <= 4; index += 1) {
    const value = yMax * index / 4;
    const y = yPosition(value);
    svg.append(svgElement("line", {class: "chart-grid", x1: margin.left, x2: width - margin.right, y1: y, y2: y}));
    const label = svgElement("text", {class: "chart-tick", x: margin.left - 10, y: y + 4, "text-anchor": "end"});
    label.textContent = metric.axis(value);
    svg.append(label);
  }

  const contexts = [...new Set(xValues)].sort((left, right) => left - right);
  const tickStep = Math.max(1, Math.ceil(contexts.length / 6));
  contexts.forEach((context, index) => {
    if (index % tickStep && index !== contexts.length - 1) return;
    const x = xPosition(context);
    svg.append(svgElement("line", {class: "chart-tick-line", x1: x, x2: x, y1: margin.top + plotHeight, y2: margin.top + plotHeight + 5}));
    const label = svgElement("text", {class: "chart-tick", x, y: height - 23, "text-anchor": "middle"});
    label.textContent = integer(context).toLocaleString();
    svg.append(label);
  });

  svg.append(svgElement("line", {class: "chart-axis", x1: margin.left, x2: width - margin.right, y1: margin.top + plotHeight, y2: margin.top + plotHeight}));
  const xLabel = svgElement("text", {class: "chart-axis-label", x: margin.left + plotWidth / 2, y: height - 3, "text-anchor": "middle"});
  xLabel.textContent = "Actual context tokens";
  svg.append(xLabel);

  const grouped = new Map();
  points.forEach(point => {
    if (!grouped.has(point.cohort.key)) grouped.set(point.cohort.key, {cohort: point.cohort, contexts: new Map()});
    const modelPoints = grouped.get(point.cohort.key).contexts;
    if (!modelPoints.has(point.context)) modelPoints.set(point.context, []);
    modelPoints.get(point.context).push(point);
  });

  [...grouped.values()].sort((left, right) => left.cohort.label.localeCompare(right.cohort.label)).forEach(({cohort, contexts: byContext}, modelIndex) => {
    const color = CHART_COLORS[modelIndex % CHART_COLORS.length];
    const lineStyle = ["", "8 4", "2 3"][Math.floor(modelIndex / CHART_COLORS.length) % 3];
    const aggregatePoints = [...byContext.entries()]
      .map(([context, matches]) => ({context, value: median(matches.map(point => point.value)), matches}))
      .sort((left, right) => left.context - right.context);
    const pathData = aggregatePoints.map((point, index) => `${index ? "L" : "M"}${xPosition(point.context)} ${yPosition(point.value)}`).join(" ");
    const path = svgElement("path", {class: "chart-series", d: pathData, stroke: color});
    if (lineStyle) path.setAttribute("stroke-dasharray", lineStyle);
    svg.append(path);
    aggregatePoints.forEach(point => {
      const accessibleLabel = `${cohort.label} / ${integer(point.context).toLocaleString()} tokens / ${metric.format(point.value)} / n=${point.matches.length}`;
      const circle = svgElement("circle", {
        class: "chart-point",
        cx: xPosition(point.context),
        cy: yPosition(point.value),
        r: 4.5,
        fill: color,
        tabindex: 0,
        role: "img",
        "aria-label": accessibleLabel,
      });
      const pointTitle = svgElement("title");
      pointTitle.textContent = accessibleLabel;
      circle.append(pointTitle);
      svg.append(circle);
    });

    const item = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.backgroundColor = "transparent";
    swatch.style.borderTop = `2px ${lineStyle === "2 3" ? "dotted" : lineStyle ? "dashed" : "solid"} ${color}`;
    item.append(swatch, document.createTextNode(cohort.label));
    legend.append(item);
  });
}

function renderComparisons() {
  renderMatrix();
  renderChart();
}

function sortedRuns() {
  return [...dashboard.runs].sort((left, right) => {
    const leftTime = epochSeconds(recordStarted(left)) || 0;
    const rightTime = epochSeconds(recordStarted(right)) || 0;
    return rightTime - leftTime;
  });
}

function runOptionLabel(record) {
  const context = recordActualContext(record);
  return `${formatTimestamp(recordStarted(record))} / ${recordModel(record)} / ${context === null ? "?" : integer(context).toLocaleString()} ctx`;
}

function renderRuns() {
  const allRecords = sortedRuns();
  const records = allRecords.slice(0, 200);
  const body = $("#run-table-body");
  const select = $("#run-select");
  const currentOption = document.createElement("option");
  currentOption.value = "";
  currentOption.textContent = "Current live run";
  const optionFragment = document.createDocumentFragment();
  optionFragment.append(currentOption);
  const rowFragment = document.createDocumentFragment();

  records.forEach(record => {
    const id = recordId(record);
    if (id) {
      const option = document.createElement("option");
      option.value = id;
      option.textContent = runOptionLabel(record);
      optionFragment.append(option);
    }

    const row = document.createElement("tr");
    if (id && id === dashboard.selectedId) row.classList.add("selected");
    const started = document.createElement("td");
    if (id) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "run-row-select";
      button.textContent = formatTimestamp(recordStarted(record));
      button.setAttribute("aria-pressed", String(id === dashboard.selectedId));
      button.addEventListener("click", () => selectRun(id));
      started.append(button);
    } else {
      started.textContent = formatTimestamp(recordStarted(record));
    }
    const values = [
      titleCase(recordPhase(record)),
      recordModel(record),
      recordActualContext(record) === null ? "--" : integer(recordActualContext(record)).toLocaleString(),
      recordOutputTokens(record) === null ? "--" : integer(recordOutputTokens(record)).toLocaleString(),
      formatDuration(recordTtft(record)),
      recordTps(record) === null ? "--" : recordTps(record).toFixed(2),
      formatDuration(recordDuration(record)),
    ];
    row.append(started);
    values.forEach(value => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    });
    rowFragment.append(row);
  });

  select.replaceChildren(optionFragment);
  select.value = dashboard.selectedId || "";
  body.replaceChildren(rowFragment);
  setText("#run-count", `${allRecords.length.toLocaleString()} run${allRecords.length === 1 ? "" : "s"}`);
  const historyCopy = dashboard.historyState === "unavailable"
    ? "Run archive endpoint unavailable. Live dashboard remains active; retrying."
    : records.length
      ? `Select a row for full sanitized measurements.${
        allRecords.length > records.length ? ` Showing the newest ${records.length}.` : ""
      }`
      : "No persisted runs yet.";
  setText("#history-state", historyCopy);
}

async function responseJson(response) {
  try {
    return await response.json();
  } catch (_error) {
    return {};
  }
}

async function loadRuns() {
  if (historyInFlight) return;
  historyInFlight = true;
  try {
    const records = [];
    const seenCursors = new Set();
    let cursor = null;
    while (true) {
      const query = new URLSearchParams({limit: String(dashboard.historyPageSize)});
      if (cursor) {
        query.set("before_started_at_ns", String(cursor.before_started_at_ns));
        query.set("before_run_id", String(cursor.before_run_id));
      }
      const response = await fetch(`/api/runs?${query}`, {cache: "no-store"});
      if (!response.ok) throw new Error(`run archive returned ${response.status}`);
      const payload = await responseJson(response);
      const page = Array.isArray(payload.runs) ? payload.runs : [];
      records.push(...page);
      cursor = payload.next_cursor || null;
      if (!cursor || page.length === 0) break;
      const key = `${cursor.before_started_at_ns}:${cursor.before_run_id}`;
      if (seenCursors.has(key)) throw new Error("run archive returned a repeated cursor");
      seenCursors.add(key);
    }
    dashboard.runs = records.filter(record => record && typeof record === "object");
    dashboard.historyState = "ready";
    if (dashboard.selectedId && !dashboard.selectedRecordComplete) {
      const summary = dashboard.runs.find(record => recordId(record) === dashboard.selectedId);
      if (summary) {
        dashboard.selectedRecord = summary;
        renderSelection(summary, true);
      }
    }
    if (dashboard.selectedId === null && snapshot) renderCurrent(snapshot, false);
  } catch (_error) {
    dashboard.historyState = "unavailable";
  } finally {
    historyInFlight = false;
    renderRuns();
    renderComparisons();
  }
}

async function selectRun(id) {
  selectionRequest += 1;
  const requestNumber = selectionRequest;
  const selectedId = id || null;
  dashboard.selectedId = selectedId;
  dashboard.selectedRecordComplete = false;
  renderRuns();

  if (!dashboard.selectedId) {
    dashboard.selectedRecord = null;
    if (snapshot) renderCurrent(snapshot);
    return;
  }

  const summary = dashboard.runs.find(record => recordId(record) === dashboard.selectedId) || {};
  dashboard.selectedRecord = summary;
  renderSelection(summary, true);
  setText("#selection-meta", `Loading run ${dashboard.selectedId}...`);
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(dashboard.selectedId)}`, {cache: "no-store"});
    if (!response.ok) throw new Error(`run detail returned ${response.status}`);
    const payload = await responseJson(response);
    if (requestNumber !== selectionRequest || dashboard.selectedId !== selectedId) return;
    const record = payload && typeof payload.record === "object" ? payload.record : payload;
    dashboard.selectedRecord = object(record);
    dashboard.selectedRecordComplete = true;
    renderSelection(dashboard.selectedRecord, true);
  } catch (_error) {
    if (requestNumber !== selectionRequest || dashboard.selectedId !== selectedId) return;
    renderSelection(summary, true);
    setText("#selection-meta", `Run ${dashboard.selectedId} / summary only; detail endpoint unavailable`);
  }
}

function adoptSnapshotRunId(run) {
  const id = recordId(run);
  if (!id) return activeProtocolKey;
  const key = `run:${id}`;
  if (key !== activeProtocolKey) renameProtocolRun(key);
  dashboard.activeRunId = id;
  return key;
}

function render(data, generation = protocolGeneration) {
  if (!data || typeof data !== "object") return;
  snapshot = data;
  const run = object(data.run);
  const otel = object(data.otel);
  const phase = String(run.phase || "starting");
  document.body.classList.toggle("running", phase === "online");
  $("#topology").classList.toggle("running", phase === "online");
  $(".status").classList.toggle("live", ["ready", "online", "refilling"].includes(phase));
  $(".status").classList.toggle("failed", phase === "error" || phase === "failed");
  setText("#status-copy", phaseText(phase));

  const inferenceRunning = Boolean(run.processes?.inference?.running);
  const preparationRunning = Boolean(run.processes?.preparation?.running);
  $("#run").disabled = phase !== "ready" || !inferenceRunning || !preparationRunning;
  $("#run").textContent = phase === "online" ? "RUNNING..." : "RUN PRIVATE CHAT";
  setText("#answer", run.text || (phase === "online" ? "Computing first token..." : "Waiting for a live run."));
  setText("#model-label", `MODEL / ${run.model_id || "unknown"}${run.tiny ? " / RANDOM-WEIGHT TRANSPORT TEST" : ""}`);
  setText("#error", run.error || "");

  if (dashboard.selectedId === null) renderCurrent(data, false);
  const runKey = adoptSnapshotRunId(run);
  renderSequence(
    otel.protocol_spans || [],
    Number(otel.protocol_cursor || 0),
    runKey,
    generation,
    Boolean(otel.protocol_truncated),
    otel.protocol_oldest_sequence,
  );

  if (lastPhase && lastPhase !== phase && ["ready", "error"].includes(phase)) loadRuns();
  lastPhase = phase;
}

async function poll() {
  if (pollInFlight) return;
  pollInFlight = true;
  const generation = protocolGeneration;
  syncProtocolGlobals();
  const cursor = protocolCursor;
  try {
    const response = await fetch(`/api/snapshot?protocol_after=${cursor}`, {cache: "no-store"});
    if (!response.ok) throw new Error(`snapshot returned ${response.status}`);
    const data = await responseJson(response);
    if (generation === protocolGeneration) render(data, generation);
  } catch (error) {
    console.error("dashboard poll failed", error);
    $(".status").classList.remove("live");
    setText("#status-copy", "COLLECTOR OFFLINE / KEEP CLI RUNNING");
  } finally {
    pollInFlight = false;
  }
}

async function run() {
  if ($("#run").disabled) {
    setText("#error", "Both protocol services must be ready before a run starts.");
    return;
  }
  const prompt = $("#prompt").value.trim();
  const maximum = Number($("#max-tokens").value);
  if (!prompt) {
    setText("#error", "Prompt is required.");
    return;
  }
  if (!Number.isInteger(maximum) || maximum < 1 || maximum > 512) {
    setText("#error", "Max output tokens must be between 1 and 512.");
    return;
  }

  const previousKey = activeProtocolKey;
  const previousActiveRunId = dashboard.activeRunId;
  const pendingRunId = `dashboard-${Date.now()}-${++runSerial}`;
  const pendingKey = `pending:${pendingRunId}`;
  startProtocolRun(pendingKey);
  dashboard.activeRunId = null;
  selectionRequest += 1;
  dashboard.selectedId = null;
  dashboard.selectedRecord = null;
  dashboard.selectedRecordComplete = false;
  renderRuns();
  showLiveSequence();
  $("#run").disabled = true;
  setText("#error", "");
  setText("#submission-status", "Starting run...");

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify({prompt, max_output_tokens: maximum, request_id: pendingRunId}),
    });
    const payload = await responseJson(response);
    if (!response.ok) {
      const rejected = new Error(String(payload.detail || payload.error || `Run failed (${response.status})`));
      rejected.rejected = true;
      throw rejected;
    }
    if (payload.run_id !== undefined && payload.run_id !== null) {
      dashboard.activeRunId = String(payload.run_id);
      renameProtocolRun(`run:${dashboard.activeRunId}`);
    }
    setText("#submission-status", `${titleCase(payload.status || "started")}${dashboard.activeRunId ? ` / ${dashboard.activeRunId}` : ""}`);
    await poll();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Run failed";
    if (error.rejected) {
      protocolGeneration += 1;
      protocolRuns.delete(pendingKey);
      activeProtocolKey = previousKey;
      dashboard.activeRunId = previousActiveRunId;
      syncProtocolGlobals();
      showLiveSequence();
      setText("#error", message);
      setText("#submission-status", "");
    } else {
      dashboard.activeRunId = pendingRunId;
      renameProtocolRun(`run:${pendingRunId}`);
      setText("#error", `${message} Reconnecting to accepted-run state.`);
      setText("#submission-status", `Reconciling / ${pendingRunId}`);
    }
    if (error.rejected && snapshot) render(snapshot);
  }
}

function updateClock() {
  const run = object(snapshot?.run);
  const started = epochSeconds(run.full_started_at ?? run.started_at);
  if (started === null) {
    setText("#run-clock", "00:00.0");
    return;
  }
  const finished = epochSeconds(run.finished_at);
  const elapsed = Math.max(0, (finished || Date.now() / 1000) - started);
  setText("#run-clock", `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${(elapsed % 60).toFixed(1).padStart(4, "0")}`);
}

async function loadConfig() {
  try {
    const response = await fetch("/api/config", {cache: "no-store"});
    if (!response.ok) return;
    const config = await responseJson(response);
    const maximum = integer(config.default_max_output_tokens, 24);
    if (maximum >= 1 && maximum <= 512) $("#max-tokens").value = String(maximum);
    const limit = integer(config.history_limit, dashboard.historyPageSize);
    if (limit >= 1) dashboard.historyPageSize = Math.min(200, limit);
  } catch (_error) {
    // The static defaults remain usable while startup is in flight.
  }
}

$("#run-form").addEventListener("submit", event => {
  event.preventDefault();
  run();
});
$("#prompt").addEventListener("keydown", event => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    run();
  }
});
$("#run-select").addEventListener("change", event => selectRun(event.target.value));
$("#metric-select").addEventListener("change", renderComparisons);

renderRuns();
renderComparisons();
clockTimer = setInterval(updateClock, 100);
loadConfig();
poll();
setInterval(poll, 500);
loadRuns();
setInterval(loadRuns, 15000);
