//! Strict native benchmarks for PLLM's compiled wrap32 region executor.

use pllm_compiler::CompiledPlan;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::fmt;
use std::hint::black_box;
use std::time::{Duration, Instant};

pub const BENCHMARK_REPORT_SCHEMA_VERSION: &str = "pllm.benchmark_report.v1";
pub const DEPLOYMENT_BENCHMARK_REPORT_SCHEMA_VERSION: &str = "pllm.deployment_benchmark_report.v1";
pub const MEASUREMENT_SCHEMA_VERSION: &str = "pllm.measurement.v1";
pub const MIN_REPETITIONS: u32 = 3;
pub const MAX_REPETITIONS: u32 = 1_000;
pub const MAX_WARMUPS: u32 = 100;
pub const MAX_THREADS: usize = 256;

const LATENCY_METRIC: &str = "latency";
const LATENCY_UNIT: &str = "seconds";
const THROUGHPUT_METRIC: &str = "throughput";
const THROUGHPUT_UNIT: &str = "region_executions_per_second";

#[derive(Clone, Debug, PartialEq)]
pub enum BenchmarkError {
    InvalidOptions(&'static str),
    InvalidPlan(String),
    Execution(String),
    OracleMismatch,
    ClockResolution,
    InvalidReport(String),
    InvalidJson(String),
}

impl fmt::Display for BenchmarkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(message) => {
                write!(formatter, "invalid benchmark options: {message}")
            }
            Self::InvalidPlan(message) => write!(formatter, "invalid compiled plan: {message}"),
            Self::Execution(message) => write!(formatter, "region execution failed: {message}"),
            Self::OracleMismatch => formatter
                .write_str("candidate output differs from scalar one-thread no-SIMD reference"),
            Self::ClockResolution => {
                formatter.write_str("timer returned zero or non-finite elapsed time")
            }
            Self::InvalidReport(message) => {
                write!(formatter, "invalid benchmark report: {message}")
            }
            Self::InvalidJson(message) => write!(formatter, "invalid benchmark JSON: {message}"),
        }
    }
}

impl std::error::Error for BenchmarkError {}

#[derive(Clone, Debug)]
pub struct BenchmarkOptions {
    pub id: String,
    pub privacy_cohort: String,
    pub numeric_cohort: String,
    pub environment: Value,
    pub warmups: u32,
    pub repetitions: u32,
    pub threads: usize,
    pub simd: bool,
}

impl BenchmarkOptions {
    pub fn validate(&self) -> Result<(), BenchmarkError> {
        if !pllm_types::valid_identity(&self.id) {
            return Err(BenchmarkError::InvalidOptions("malformed benchmark id"));
        }
        if !pllm_types::valid_identity(&self.privacy_cohort) {
            return Err(BenchmarkError::InvalidOptions("malformed privacy cohort"));
        }
        if !pllm_types::valid_identity(&self.numeric_cohort) {
            return Err(BenchmarkError::InvalidOptions("malformed numeric cohort"));
        }
        if !matches!(&self.environment, Value::Object(values) if !values.is_empty()) {
            return Err(BenchmarkError::InvalidOptions(
                "environment must be a non-empty JSON object",
            ));
        }
        if self.warmups > MAX_WARMUPS {
            return Err(BenchmarkError::InvalidOptions(
                "warmups must be between 0 and 100",
            ));
        }
        if !(MIN_REPETITIONS..=MAX_REPETITIONS).contains(&self.repetitions) {
            return Err(BenchmarkError::InvalidOptions(
                "repetitions must be between 3 and 1000",
            ));
        }
        if !(1..=MAX_THREADS).contains(&self.threads) {
            return Err(BenchmarkError::InvalidOptions(
                "threads must be between 1 and 256",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementOrigin {
    NativeExecuted,
    ImportedArchive,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementScope {
    Region,
    Deployment,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementPhase {
    Offline,
    Online,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimStatus {
    NotClaimed,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SampleOrder {
    ScalarThenCandidate,
    CandidateThenScalar,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Measurement {
    pub schema_version: String,
    pub id: String,
    pub origin: MeasurementOrigin,
    pub scope: MeasurementScope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<MeasurementPhase>,
    pub metric: String,
    pub privacy_cohort: String,
    pub numeric_cohort: String,
    pub value: f64,
    pub unit: String,
    pub unavailable_reason: Option<String>,
    pub plan_lock_digest: Option<String>,
    pub environment_digest: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_record_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_paths: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionMeasurements {
    pub latency: Measurement,
    pub throughput: Measurement,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkSample {
    pub index: u32,
    pub order: SampleOrder,
    pub scalar_reference: ExecutionMeasurements,
    pub candidate: ExecutionMeasurements,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkParameters {
    pub warmups: u32,
    pub repetitions: u32,
    pub candidate_threads: usize,
    pub candidate_simd: bool,
    pub reference_threads: usize,
    pub reference_simd: bool,
    pub batch: u64,
    pub input_width: u64,
    pub output_width: u64,
    pub weight_elements: usize,
    pub input_elements: usize,
    pub output_elements: usize,
    pub weight_type: String,
    pub numeric_type: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimBoundary {
    pub full_model: ClaimStatus,
    pub privacy: ClaimStatus,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkReport {
    pub schema_version: String,
    pub id: String,
    pub origin: MeasurementOrigin,
    pub scope: MeasurementScope,
    pub plan_lock_digest: String,
    pub environment_digest: String,
    pub environment: Value,
    pub privacy_cohort: String,
    pub numeric_cohort: String,
    pub parameters: BenchmarkParameters,
    pub output_digest: String,
    pub claim_boundary: ClaimBoundary,
    pub samples: Vec<BenchmarkSample>,
}

impl BenchmarkReport {
    pub fn validate(&self) -> Result<(), BenchmarkError> {
        if self.schema_version != BENCHMARK_REPORT_SCHEMA_VERSION {
            return invalid_report("wrong report schema version");
        }
        if self.origin != MeasurementOrigin::NativeExecuted
            || self.scope != MeasurementScope::Region
        {
            return invalid_report("invalid benchmark origin or scope");
        }
        if !pllm_types::valid_identity(&self.id) {
            return invalid_report("malformed benchmark id");
        }
        if !pllm_types::valid_identity(&self.privacy_cohort)
            || !pllm_types::valid_identity(&self.numeric_cohort)
        {
            return invalid_report("malformed cohort");
        }
        if !matches!(&self.environment, Value::Object(values) if !values.is_empty()) {
            return invalid_report("environment must be a non-empty JSON object");
        }
        validate_digest(&self.plan_lock_digest)?;
        validate_digest(&self.environment_digest)?;
        validate_digest(&self.output_digest)?;
        if sha256_hex(&pllm_types::canonical_bytes(&self.environment)) != self.environment_digest {
            return invalid_report("environment digest mismatch");
        }
        validate_parameters(&self.parameters)?;
        if self.samples.len() != self.parameters.repetitions as usize {
            return invalid_report("sample count differs from repetitions");
        }

        for (index, sample) in self.samples.iter().enumerate() {
            let index = index as u32;
            if sample.index != index || sample.order != order_for(index) {
                return invalid_report("sample index or execution order mismatch");
            }
            validate_measurements(&sample.scalar_reference, "scalar", index, self)?;
            validate_measurements(&sample.candidate, "candidate", index, self)?;
        }
        Ok(())
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>, BenchmarkError> {
        self.validate()?;
        Ok(pllm_types::canonical_bytes(self))
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self, BenchmarkError> {
        let report: Self = serde_json::from_slice(bytes)
            .map_err(|error| BenchmarkError::InvalidJson(error.to_string()))?;
        report.validate()?;
        Ok(report)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentReportOptions {
    pub id: String,
    pub plan_lock_digest: Option<String>,
    pub privacy_cohort: String,
    pub numeric_cohort: String,
    pub environment: Value,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentObservation {
    pub role: String,
    pub phase: MeasurementPhase,
    pub origin: MeasurementOrigin,
    pub metric: String,
    pub unit: String,
    pub value: f64,
    #[serde(default)]
    pub notes: Option<String>,
    #[serde(default)]
    pub evidence_paths: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentBenchmarkRequest {
    pub options: DeploymentReportOptions,
    pub observations: Vec<DeploymentObservation>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentBenchmarkReport {
    pub schema_version: String,
    pub id: String,
    pub plan_lock_digest: Option<String>,
    pub environment_digest: String,
    pub environment: Value,
    pub privacy_cohort: String,
    pub numeric_cohort: String,
    pub measurements: Vec<Measurement>,
}

impl DeploymentBenchmarkReport {
    pub fn validate(&self) -> Result<(), BenchmarkError> {
        if self.schema_version != DEPLOYMENT_BENCHMARK_REPORT_SCHEMA_VERSION {
            return invalid_report("wrong deployment report schema version");
        }
        validate_deployment_options(&DeploymentReportOptions {
            id: self.id.clone(),
            plan_lock_digest: self.plan_lock_digest.clone(),
            privacy_cohort: self.privacy_cohort.clone(),
            numeric_cohort: self.numeric_cohort.clone(),
            environment: self.environment.clone(),
        })
        .map_err(|error| BenchmarkError::InvalidReport(error.to_string()))?;
        validate_digest(&self.environment_digest)?;
        if sha256_hex(&pllm_types::canonical_bytes(&self.environment)) != self.environment_digest {
            return invalid_report("environment digest mismatch");
        }
        if self.measurements.is_empty() {
            return invalid_report("deployment report must contain observations");
        }
        for (index, measurement) in self.measurements.iter().enumerate() {
            validate_deployment_measurement(measurement, index, self)?;
        }
        Ok(())
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>, BenchmarkError> {
        self.validate()?;
        Ok(pllm_types::canonical_bytes(self))
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self, BenchmarkError> {
        let report: Self = serde_json::from_slice(bytes)
            .map_err(|error| BenchmarkError::InvalidJson(error.to_string()))?;
        report.validate()?;
        Ok(report)
    }
}

pub fn deployment_report(
    request: DeploymentBenchmarkRequest,
) -> Result<DeploymentBenchmarkReport, BenchmarkError> {
    validate_deployment_options(&request.options)?;
    if request.observations.is_empty() {
        return Err(BenchmarkError::InvalidOptions(
            "observations must not be empty",
        ));
    }

    let environment_digest = sha256_hex(&pllm_types::canonical_bytes(&request.options.environment));
    let mut measurements = Vec::with_capacity(request.observations.len());
    for (index, observation) in request.observations.into_iter().enumerate() {
        validate_deployment_observation(&observation)?;
        measurements.push(Measurement {
            schema_version: MEASUREMENT_SCHEMA_VERSION.into(),
            id: format!("{}.observation.{index:04}", request.options.id),
            origin: observation.origin,
            scope: MeasurementScope::Deployment,
            role: Some(observation.role),
            phase: Some(observation.phase),
            metric: observation.metric,
            privacy_cohort: request.options.privacy_cohort.clone(),
            numeric_cohort: request.options.numeric_cohort.clone(),
            value: observation.value,
            unit: observation.unit,
            unavailable_reason: None,
            plan_lock_digest: request.options.plan_lock_digest.clone(),
            environment_digest: environment_digest.clone(),
            source_record_ids: Vec::new(),
            notes: observation.notes,
            evidence_paths: observation.evidence_paths,
        });
    }

    let report = DeploymentBenchmarkReport {
        schema_version: DEPLOYMENT_BENCHMARK_REPORT_SCHEMA_VERSION.into(),
        id: request.options.id,
        plan_lock_digest: request.options.plan_lock_digest,
        environment_digest,
        environment: request.options.environment,
        privacy_cohort: request.options.privacy_cohort,
        numeric_cohort: request.options.numeric_cohort,
        measurements,
    };
    report.validate()?;
    Ok(report)
}

pub fn benchmark_wrap32(
    compiled: &CompiledPlan,
    weights: &[i8],
    input: &[u32],
    options: BenchmarkOptions,
) -> Result<BenchmarkReport, BenchmarkError> {
    benchmark_with(
        compiled,
        weights,
        input,
        options,
        |plan, weights, input, threads, simd| {
            pllm_compiler::execute_wrap32(plan, weights, input, threads, simd)
        },
    )
}

fn benchmark_with<F>(
    compiled: &CompiledPlan,
    weights: &[i8],
    input: &[u32],
    options: BenchmarkOptions,
    execute: F,
) -> Result<BenchmarkReport, BenchmarkError>
where
    F: Fn(&CompiledPlan, &[u8], &[u32], usize, bool) -> Result<Vec<u32>, String>,
{
    options.validate()?;
    compiled.verify().map_err(BenchmarkError::InvalidPlan)?;
    let encoded_weights: Vec<u8> = weights.iter().map(|weight| *weight as u8).collect();

    let reference = invoke(&execute, compiled, &encoded_weights, input, 1, false)?;
    let candidate = invoke(
        &execute,
        compiled,
        &encoded_weights,
        input,
        options.threads,
        options.simd,
    )?;
    if candidate != reference {
        return Err(BenchmarkError::OracleMismatch);
    }

    for index in 0..options.warmups {
        run_pair(
            order_for(index),
            &execute,
            compiled,
            &encoded_weights,
            input,
            &options,
            &reference,
            false,
        )?;
    }

    let plan_lock_digest = pllm_types::plan_lock_digest(&compiled.lock).to_string();
    let environment_digest = sha256_hex(&pllm_types::canonical_bytes(&options.environment));
    let dimensions = dimensions(compiled)?;
    let mut samples = Vec::with_capacity(options.repetitions as usize);
    for index in 0..options.repetitions {
        let order = order_for(index);
        let (scalar, candidate) = run_pair(
            order,
            &execute,
            compiled,
            &encoded_weights,
            input,
            &options,
            &reference,
            true,
        )?;
        samples.push(BenchmarkSample {
            index,
            order,
            scalar_reference: measurements(
                "scalar",
                index,
                scalar.ok_or(BenchmarkError::ClockResolution)?,
                &options,
                &plan_lock_digest,
                &environment_digest,
            )?,
            candidate: measurements(
                "candidate",
                index,
                candidate.ok_or(BenchmarkError::ClockResolution)?,
                &options,
                &plan_lock_digest,
                &environment_digest,
            )?,
        });
    }

    let report = BenchmarkReport {
        schema_version: BENCHMARK_REPORT_SCHEMA_VERSION.into(),
        id: options.id.clone(),
        origin: MeasurementOrigin::NativeExecuted,
        scope: MeasurementScope::Region,
        plan_lock_digest,
        environment_digest,
        environment: options.environment.clone(),
        privacy_cohort: options.privacy_cohort.clone(),
        numeric_cohort: options.numeric_cohort.clone(),
        parameters: BenchmarkParameters {
            warmups: options.warmups,
            repetitions: options.repetitions,
            candidate_threads: options.threads,
            candidate_simd: options.simd,
            reference_threads: 1,
            reference_simd: false,
            batch: dimensions.0,
            input_width: dimensions.1,
            output_width: dimensions.2,
            weight_elements: weights.len(),
            input_elements: input.len(),
            output_elements: reference.len(),
            weight_type: "i8_twos_complement".into(),
            numeric_type: "wrap32".into(),
        },
        output_digest: output_digest(&reference),
        claim_boundary: ClaimBoundary {
            full_model: ClaimStatus::NotClaimed,
            privacy: ClaimStatus::NotClaimed,
        },
        samples,
    };
    report.validate()?;
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
fn run_pair<F>(
    order: SampleOrder,
    execute: &F,
    compiled: &CompiledPlan,
    weights: &[u8],
    input: &[u32],
    options: &BenchmarkOptions,
    expected: &[u32],
    timed: bool,
) -> Result<(Option<Duration>, Option<Duration>), BenchmarkError>
where
    F: Fn(&CompiledPlan, &[u8], &[u32], usize, bool) -> Result<Vec<u32>, String>,
{
    let scalar = || run_once(execute, compiled, weights, input, 1, false, expected, timed);
    let candidate = || {
        run_once(
            execute,
            compiled,
            weights,
            input,
            options.threads,
            options.simd,
            expected,
            timed,
        )
    };
    match order {
        SampleOrder::ScalarThenCandidate => Ok((scalar()?, candidate()?)),
        SampleOrder::CandidateThenScalar => {
            let candidate = candidate()?;
            let scalar = scalar()?;
            Ok((scalar, candidate))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_once<F>(
    execute: &F,
    compiled: &CompiledPlan,
    weights: &[u8],
    input: &[u32],
    threads: usize,
    simd: bool,
    expected: &[u32],
    timed: bool,
) -> Result<Option<Duration>, BenchmarkError>
where
    F: Fn(&CompiledPlan, &[u8], &[u32], usize, bool) -> Result<Vec<u32>, String>,
{
    let start = timed.then(Instant::now);
    let output = invoke(execute, compiled, weights, input, threads, simd)?;
    let elapsed = start.map(|start| start.elapsed());
    black_box(&output);
    if output != expected {
        return Err(BenchmarkError::OracleMismatch);
    }
    Ok(elapsed)
}

fn invoke<F>(
    execute: &F,
    compiled: &CompiledPlan,
    weights: &[u8],
    input: &[u32],
    threads: usize,
    simd: bool,
) -> Result<Vec<u32>, BenchmarkError>
where
    F: Fn(&CompiledPlan, &[u8], &[u32], usize, bool) -> Result<Vec<u32>, String>,
{
    black_box(execute(
        black_box(compiled),
        black_box(weights),
        black_box(input),
        black_box(threads),
        black_box(simd),
    ))
    .map_err(BenchmarkError::Execution)
}

fn measurements(
    variant: &str,
    index: u32,
    elapsed: Duration,
    options: &BenchmarkOptions,
    plan_lock_digest: &str,
    environment_digest: &str,
) -> Result<ExecutionMeasurements, BenchmarkError> {
    let seconds = elapsed.as_secs_f64();
    if seconds <= 0.0 || !seconds.is_finite() {
        return Err(BenchmarkError::ClockResolution);
    }
    let throughput = seconds.recip();
    if !throughput.is_finite() {
        return Err(BenchmarkError::ClockResolution);
    }
    let measurement = |metric: &str, value: f64, unit: &str| Measurement {
        schema_version: MEASUREMENT_SCHEMA_VERSION.into(),
        id: measurement_id(index, variant, metric),
        origin: MeasurementOrigin::NativeExecuted,
        scope: MeasurementScope::Region,
        role: None,
        phase: None,
        metric: metric.into(),
        privacy_cohort: options.privacy_cohort.clone(),
        numeric_cohort: options.numeric_cohort.clone(),
        value,
        unit: unit.into(),
        unavailable_reason: None,
        plan_lock_digest: Some(plan_lock_digest.into()),
        environment_digest: environment_digest.into(),
        source_record_ids: Vec::new(),
        notes: None,
        evidence_paths: Vec::new(),
    };
    Ok(ExecutionMeasurements {
        latency: measurement(LATENCY_METRIC, seconds, LATENCY_UNIT),
        throughput: measurement(THROUGHPUT_METRIC, throughput, THROUGHPUT_UNIT),
    })
}

fn validate_parameters(parameters: &BenchmarkParameters) -> Result<(), BenchmarkError> {
    let expected_weights = element_count(parameters.output_width, parameters.input_width);
    let expected_input = element_count(parameters.batch, parameters.input_width);
    let expected_output = element_count(parameters.batch, parameters.output_width);
    if parameters.warmups > MAX_WARMUPS
        || !(MIN_REPETITIONS..=MAX_REPETITIONS).contains(&parameters.repetitions)
        || !(1..=MAX_THREADS).contains(&parameters.candidate_threads)
        || parameters.reference_threads != 1
        || parameters.reference_simd
        || parameters.batch == 0
        || parameters.input_width == 0
        || parameters.output_width == 0
        || parameters.weight_elements == 0
        || parameters.input_elements == 0
        || parameters.output_elements == 0
        || expected_weights != Some(parameters.weight_elements)
        || expected_input != Some(parameters.input_elements)
        || expected_output != Some(parameters.output_elements)
        || parameters.weight_type != "i8_twos_complement"
        || parameters.numeric_type != "wrap32"
    {
        return invalid_report("invalid benchmark parameters");
    }
    Ok(())
}

fn element_count(left: u64, right: u64) -> Option<usize> {
    usize::try_from(left.checked_mul(right)?).ok()
}

fn validate_measurements(
    measurements: &ExecutionMeasurements,
    variant: &str,
    index: u32,
    report: &BenchmarkReport,
) -> Result<(), BenchmarkError> {
    validate_measurement(
        &measurements.latency,
        variant,
        index,
        LATENCY_METRIC,
        LATENCY_UNIT,
        report,
    )?;
    validate_measurement(
        &measurements.throughput,
        variant,
        index,
        THROUGHPUT_METRIC,
        THROUGHPUT_UNIT,
        report,
    )?;
    let expected_throughput = measurements.latency.value.recip();
    let reciprocal_error = (measurements.throughput.value - expected_throughput).abs();
    if reciprocal_error > expected_throughput.abs() * f64::EPSILON * 4.0 {
        return invalid_report("throughput is not reciprocal of latency");
    }
    Ok(())
}

fn validate_measurement(
    measurement: &Measurement,
    variant: &str,
    index: u32,
    metric: &str,
    unit: &str,
    report: &BenchmarkReport,
) -> Result<(), BenchmarkError> {
    if measurement.schema_version != MEASUREMENT_SCHEMA_VERSION
        || measurement.id != measurement_id(index, variant, metric)
        || measurement.origin != MeasurementOrigin::NativeExecuted
        || measurement.scope != MeasurementScope::Region
        || measurement.role.is_some()
        || measurement.phase.is_some()
        || measurement.metric != metric
        || measurement.unit != unit
        || measurement.privacy_cohort != report.privacy_cohort
        || measurement.numeric_cohort != report.numeric_cohort
        || measurement.plan_lock_digest.as_deref() != Some(report.plan_lock_digest.as_str())
        || measurement.environment_digest != report.environment_digest
        || measurement.unavailable_reason.is_some()
        || !measurement.source_record_ids.is_empty()
        || measurement.notes.is_some()
        || !measurement.evidence_paths.is_empty()
        || !measurement.value.is_finite()
        || measurement.value <= 0.0
    {
        return invalid_report("invalid measurement");
    }
    Ok(())
}

fn validate_deployment_options(options: &DeploymentReportOptions) -> Result<(), BenchmarkError> {
    if !pllm_types::valid_identity(&options.id) {
        return Err(BenchmarkError::InvalidOptions(
            "malformed deployment benchmark id",
        ));
    }
    if !pllm_types::valid_identity(&options.privacy_cohort)
        || !pllm_types::valid_identity(&options.numeric_cohort)
    {
        return Err(BenchmarkError::InvalidOptions(
            "malformed deployment benchmark cohort",
        ));
    }
    if !matches!(&options.environment, Value::Object(values) if !values.is_empty()) {
        return Err(BenchmarkError::InvalidOptions(
            "environment must be a non-empty JSON object",
        ));
    }
    if let Some(digest) = &options.plan_lock_digest {
        validate_digest(digest)
            .map_err(|_| BenchmarkError::InvalidOptions("malformed plan lock digest"))?;
    }
    Ok(())
}

fn validate_deployment_observation(
    observation: &DeploymentObservation,
) -> Result<(), BenchmarkError> {
    if !pllm_types::valid_identity(&observation.role) {
        return Err(BenchmarkError::InvalidOptions(
            "observation role must be a safe non-empty identifier",
        ));
    }
    if !pllm_types::valid_identity(&observation.metric)
        || !pllm_types::valid_identity(&observation.unit)
    {
        return Err(BenchmarkError::InvalidOptions(
            "observation metric and unit must be measurement identifiers",
        ));
    }
    if !observation.value.is_finite() || observation.value < 0.0 {
        return Err(BenchmarkError::InvalidOptions(
            "observation value must be finite and non-negative",
        ));
    }
    if observation.notes.as_ref().is_some_and(String::is_empty) {
        return Err(BenchmarkError::InvalidOptions(
            "observation notes must not be empty",
        ));
    }
    for (index, path) in observation.evidence_paths.iter().enumerate() {
        if path.is_empty() || observation.evidence_paths[..index].contains(path) {
            return Err(BenchmarkError::InvalidOptions(
                "observation evidence paths must be non-empty and unique",
            ));
        }
    }
    Ok(())
}

fn validate_deployment_measurement(
    measurement: &Measurement,
    index: usize,
    report: &DeploymentBenchmarkReport,
) -> Result<(), BenchmarkError> {
    if measurement.schema_version != MEASUREMENT_SCHEMA_VERSION
        || measurement.id != format!("{}.observation.{index:04}", report.id)
        || measurement.scope != MeasurementScope::Deployment
        || measurement.role.is_none()
        || measurement.phase.is_none()
        || measurement.privacy_cohort != report.privacy_cohort
        || measurement.numeric_cohort != report.numeric_cohort
        || measurement.plan_lock_digest != report.plan_lock_digest
        || measurement.environment_digest != report.environment_digest
        || measurement.unavailable_reason.is_some()
        || !measurement.source_record_ids.is_empty()
    {
        return invalid_report("invalid deployment measurement");
    }
    validate_deployment_observation(&DeploymentObservation {
        role: measurement.role.clone().expect("role checked above"),
        phase: measurement.phase.expect("phase checked above"),
        origin: measurement.origin,
        metric: measurement.metric.clone(),
        unit: measurement.unit.clone(),
        value: measurement.value,
        notes: measurement.notes.clone(),
        evidence_paths: measurement.evidence_paths.clone(),
    })
    .map_err(|error| BenchmarkError::InvalidReport(error.to_string()))
}

fn dimensions(compiled: &CompiledPlan) -> Result<(u64, u64, u64), BenchmarkError> {
    let [step] = compiled.region_program.steps.as_slice() else {
        return Err(BenchmarkError::InvalidPlan(
            "wrap32 benchmark requires exactly one region step".into(),
        ));
    };
    let ([batch, input_width], [output_batch, output_width]) =
        (step.input.shape.as_slice(), step.output.shape.as_slice())
    else {
        return Err(BenchmarkError::InvalidPlan(
            "wrap32 benchmark requires rank-two input and output".into(),
        ));
    };
    if batch != output_batch {
        return Err(BenchmarkError::InvalidPlan(
            "input and output batch dimensions differ".into(),
        ));
    }
    Ok((*batch, *input_width, *output_width))
}

fn order_for(index: u32) -> SampleOrder {
    if index % 2 == 0 {
        SampleOrder::ScalarThenCandidate
    } else {
        SampleOrder::CandidateThenScalar
    }
}

fn measurement_id(index: u32, variant: &str, metric: &str) -> String {
    format!("benchmark.sample.{index:04}.{variant}.{metric}")
}

fn output_digest(output: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(output.len() * 4);
    for value in output {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    sha256_hex(&bytes)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in digest {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn validate_digest(value: &str) -> Result<(), BenchmarkError> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        invalid_report("malformed digest")
    }
}

fn invalid_report<T>(message: &str) -> Result<T, BenchmarkError> {
    Err(BenchmarkError::InvalidReport(message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compiled_plan() -> CompiledPlan {
        pllm_compiler::compile_document(include_bytes!(
            "../../../schemas/fixtures/compile-request.valid.json"
        ))
        .expect("public compiler fixture must compile")
    }

    fn options() -> BenchmarkOptions {
        BenchmarkOptions {
            id: "bench.fixture".into(),
            privacy_cohort: "privacy.single-evaluator".into(),
            numeric_cohort: "numeric.wrap32".into(),
            environment: serde_json::json!({
                "cpu": "fixture",
                "os": "fixture",
            }),
            warmups: 1,
            repetitions: 3,
            threads: 2,
            simd: true,
        }
    }

    fn inputs() -> (Vec<i8>, Vec<u32>) {
        (vec![-1, 2, -3, 4, -5, 6], vec![1, 2, 3, u32::MAX, 5, 6])
    }

    fn make_report() -> BenchmarkReport {
        let (weights, input) = inputs();
        benchmark_wrap32(&compiled_plan(), &weights, &input, options()).unwrap()
    }

    fn deployment_request() -> DeploymentBenchmarkRequest {
        DeploymentBenchmarkRequest {
            options: DeploymentReportOptions {
                id: "deployment.fixture".into(),
                plan_lock_digest: None,
                privacy_cohort: "privacy.single-evaluator".into(),
                numeric_cohort: "numeric.wrap32".into(),
                environment: serde_json::json!({
                    "cpu": "fixture",
                    "transport": "loopback",
                }),
            },
            observations: vec![
                DeploymentObservation {
                    role: "client".into(),
                    phase: MeasurementPhase::Online,
                    origin: MeasurementOrigin::NativeExecuted,
                    metric: "latency".into(),
                    unit: "seconds".into(),
                    value: 1.25,
                    notes: Some("cold run".into()),
                    evidence_paths: vec!["evidence/run.json".into()],
                },
                DeploymentObservation {
                    role: "preparation".into(),
                    phase: MeasurementPhase::Offline,
                    origin: MeasurementOrigin::ImportedArchive,
                    metric: "throughput".into(),
                    unit: "tokens_per_second".into(),
                    value: 0.0,
                    notes: None,
                    evidence_paths: Vec::new(),
                },
            ],
        }
    }

    #[test]
    fn native_report_is_canonical_and_deterministically_linked() {
        let report = make_report();
        let bytes = report.canonical_json().unwrap();
        assert_eq!(bytes, report.canonical_json().unwrap());
        BenchmarkReport::from_json(&bytes).unwrap();
        assert_eq!(report.origin, MeasurementOrigin::NativeExecuted);
        assert_eq!(report.scope, MeasurementScope::Region);
        assert_eq!(
            report.plan_lock_digest,
            pllm_types::plan_lock_digest(&compiled_plan().lock).as_str()
        );
        assert_eq!(report.claim_boundary.full_model, ClaimStatus::NotClaimed);
        assert_eq!(report.claim_boundary.privacy, ClaimStatus::NotClaimed);
        assert_eq!(report.samples.len(), 3);
        assert_eq!(report.samples[0].order, SampleOrder::ScalarThenCandidate);
        assert_eq!(report.samples[1].order, SampleOrder::CandidateThenScalar);
        assert_eq!(
            report.environment_digest,
            sha256_hex(&pllm_types::canonical_bytes(&serde_json::json!({
                "os": "fixture",
                "cpu": "fixture",
            })))
        );
    }

    #[test]
    fn validates_bounds_id_cohorts_and_environment() {
        let mut value = options();
        value.repetitions = 2;
        assert!(value.validate().is_err());
        value = options();
        value.warmups = 101;
        assert!(value.validate().is_err());
        value = options();
        value.threads = 0;
        assert!(value.validate().is_err());
        value = options();
        value.id = "bad id".into();
        assert!(value.validate().is_err());
        value = options();
        value.privacy_cohort.clear();
        assert!(value.validate().is_err());
        value = options();
        value.numeric_cohort = "bad cohort".into();
        assert!(value.validate().is_err());
        value = options();
        value.environment = serde_json::json!({});
        assert!(value.validate().is_err());
    }

    #[test]
    fn rejects_candidate_oracle_mismatch_before_sampling() {
        let plan = compiled_plan();
        let (weights, input) = inputs();
        let result = benchmark_with(
            &plan,
            &weights,
            &input,
            options(),
            |plan, weights, input, threads, simd| {
                let mut output =
                    pllm_compiler::execute_wrap32(plan, weights, input, threads, simd)?;
                if threads != 1 || simd {
                    output[0] = output[0].wrapping_add(1);
                }
                Ok(output)
            },
        );
        assert_eq!(result.unwrap_err(), BenchmarkError::OracleMismatch);
    }

    #[test]
    fn measurements_match_schema_semantics_and_reject_nonfinite_values() {
        let mut report = make_report();
        for sample in &report.samples {
            for values in [&sample.scalar_reference, &sample.candidate] {
                assert_eq!(values.latency.metric, "latency");
                assert_eq!(values.latency.unit, "seconds");
                assert_eq!(values.throughput.metric, "throughput");
                assert_eq!(values.throughput.unit, "region_executions_per_second");
                assert_eq!(values.throughput.value, values.latency.value.recip());
                let json = serde_json::to_value(&values.latency).unwrap();
                for required in [
                    "schema_version",
                    "id",
                    "origin",
                    "scope",
                    "metric",
                    "privacy_cohort",
                    "numeric_cohort",
                    "value",
                    "unit",
                    "unavailable_reason",
                    "plan_lock_digest",
                    "environment_digest",
                ] {
                    assert!(json.get(required).is_some(), "missing {required}");
                }
            }
        }
        report.samples[0].candidate.latency.value = f64::NAN;
        assert!(report.validate().is_err());
    }

    #[test]
    fn report_validation_detects_digest_and_measurement_tampering() {
        let mut report = make_report();
        report.environment["cpu"] = Value::String("other".into());
        assert!(report.validate().is_err());

        let mut report = make_report();
        report.samples[0].candidate.throughput.value *= 2.0;
        assert!(report.validate().is_err());

        let mut report = make_report();
        report.parameters.weight_elements += 1;
        assert!(report.validate().is_err());

        let mut report = make_report();
        report.samples[0].candidate.latency.plan_lock_digest = Some("0".repeat(64));
        assert!(report.validate().is_err());
    }

    #[test]
    fn deployment_report_is_canonical_with_one_measurement_per_observation() {
        let report = deployment_report(deployment_request()).unwrap();
        let digest = "d5b5136512001b6c947982c49b5960becc3db03bbd0a5016bb2a7ce830808ff6";
        let expected = format!(
            concat!(
                "{{\"environment\":{{\"cpu\":\"fixture\",\"transport\":\"loopback\"}},",
                "\"environment_digest\":\"{digest}\",\"id\":\"deployment.fixture\",",
                "\"measurements\":[{{\"environment_digest\":\"{digest}\",",
                "\"evidence_paths\":[\"evidence/run.json\"],",
                "\"id\":\"deployment.fixture.observation.0000\",\"metric\":\"latency\",",
                "\"notes\":\"cold run\",\"numeric_cohort\":\"numeric.wrap32\",",
                "\"origin\":\"native_executed\",\"phase\":\"online\",",
                "\"plan_lock_digest\":null,\"privacy_cohort\":\"privacy.single-evaluator\",",
                "\"role\":\"client\",\"schema_version\":\"pllm.measurement.v1\",",
                "\"scope\":\"deployment\",\"unavailable_reason\":null,",
                "\"unit\":\"seconds\",\"value\":1.25}},",
                "{{\"environment_digest\":\"{digest}\",",
                "\"id\":\"deployment.fixture.observation.0001\",\"metric\":\"throughput\",",
                "\"numeric_cohort\":\"numeric.wrap32\",\"origin\":\"imported_archive\",",
                "\"phase\":\"offline\",\"plan_lock_digest\":null,",
                "\"privacy_cohort\":\"privacy.single-evaluator\",\"role\":\"preparation\",",
                "\"schema_version\":\"pllm.measurement.v1\",\"scope\":\"deployment\",",
                "\"unavailable_reason\":null,\"unit\":\"tokens_per_second\",",
                "\"value\":0.0}}],\"numeric_cohort\":\"numeric.wrap32\",",
                "\"plan_lock_digest\":null,\"privacy_cohort\":\"privacy.single-evaluator\",",
                "\"schema_version\":\"pllm.deployment_benchmark_report.v1\"}}"
            ),
            digest = digest,
        );
        let bytes = report.canonical_json().unwrap();
        assert_eq!(bytes, expected.as_bytes());
        assert_eq!(report.measurements.len(), 2);
        DeploymentBenchmarkReport::from_json(&bytes).unwrap();
    }

    #[test]
    fn deployment_request_serde_and_values_are_strict() {
        let mut value = serde_json::to_value(deployment_request()).unwrap();
        value["extra"] = serde_json::json!(true);
        assert!(serde_json::from_value::<DeploymentBenchmarkRequest>(value).is_err());

        let mut value = serde_json::to_value(deployment_request()).unwrap();
        value["observations"][0]["phase"] = serde_json::json!("transition");
        assert!(serde_json::from_value::<DeploymentBenchmarkRequest>(value).is_err());

        let mut value = serde_json::to_value(deployment_request()).unwrap();
        value["options"]
            .as_object_mut()
            .unwrap()
            .remove("plan_lock_digest");
        assert!(serde_json::from_value::<DeploymentBenchmarkRequest>(value).is_ok());

        let mut request = deployment_request();
        request.observations[0].role = "bad role".into();
        assert!(deployment_report(request).is_err());
        let mut request = deployment_request();
        request.observations[0].value = f64::NAN;
        assert!(deployment_report(request).is_err());
        let mut request = deployment_request();
        request.observations[0].value = -0.1;
        assert!(deployment_report(request).is_err());
        let mut request = deployment_request();
        request.observations.clear();
        assert!(deployment_report(request).is_err());
        let mut request = deployment_request();
        request.options.plan_lock_digest = Some("not-a-digest".into());
        assert!(deployment_report(request).is_err());
    }
}
