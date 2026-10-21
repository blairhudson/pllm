//! Separate-process conformance using bounded test-only pipes, not a core server
//! or a claim of independent operators / authenticated network deployment.
use std::{
    io::{BufRead, BufReader, Read, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::Arc,
};

use pllm_core::gkm::*;

const ID: [u8; 32] = [41; 32];

fn sources() -> (I16Matrix, FloatTable) {
    let matrix = I16Matrix::from_encoded(1, 2, &[3, -7], &[4096; 2]).unwrap();
    let rows: Vec<u8> = [0.25f32, 0.5, 1.5, -0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    (matrix, FloatTable::from_le_bytes(2, 2, &rows).unwrap())
}

fn read_packet(input: &mut impl Read, limit: usize) -> Vec<u8> {
    let mut length = [0u8; 4];
    input.read_exact(&mut length).unwrap();
    let length = u32::from_le_bytes(length) as usize;
    assert!(length <= limit);
    let mut bytes = vec![0; length];
    input.read_exact(&mut bytes).unwrap();
    bytes
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        write!(out, "{byte:02x}").unwrap();
    }
    out
}

fn unhex(bytes: &str) -> Vec<u8> {
    assert!(bytes.len() <= MAX_WIRE_FRAME_BYTES * 2);
    bytes
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

#[test]
fn worker_process_fixture() {
    let Ok(role) = std::env::var("PLLM_GKM_WIRE_CHILD") else {
        return;
    };
    let role = match role.as_str() {
        "0" => Worker::Zero,
        "1" => Worker::One,
        _ => panic!("bad test role"),
    };
    let mut input = std::io::stdin().lock();
    // The parent supplies the separately pinned identity through its trusted
    // fixture pipe. The child receives no other role's start/share or client data.
    let mut pin = [0u8; 32];
    input.read_exact(&mut pin).unwrap();
    let publication = read_packet(&mut input, MAX_PUBLICATION_WIRE_BYTES);
    let public = Arc::new(Publication::import_wire(&publication, pin).unwrap());
    let (matrix, table) = sources();
    let mut worker = WorkerWire::new(
        public,
        pin,
        role,
        &[Stage {
            id: ID,
            matrix: &matrix,
        }],
        &table,
    )
    .unwrap();
    let start = read_packet(&mut input, MAX_WIRE_FRAME_BYTES);
    let seeded = start[9] == 1;
    assert_eq!(
        start.len(),
        296 + 2 * 33 + if seeded && role == Worker::One { 32 } else { 0 }
    );
    let mut window = worker.begin(&start).unwrap();
    for _ in 0..2 {
        let request = read_packet(&mut input, MAX_WIRE_FRAME_BYTES);
        if seeded && role == Worker::One {
            assert_eq!(request.len(), 296);
        }
        let reply = window.execute(&request).unwrap();
        println!("WIRE_REPLY={}", hex(reply.as_bytes()));
        std::io::stdout().flush().unwrap();
    }
    assert_eq!(window.finish().unwrap().worker, role);
}

struct Process {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}

impl Process {
    fn start(role: &str, public: &Publication, packet: &WireBytes, start: &WireBytes) -> Self {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "worker_process_fixture",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("PLLM_GKM_WIRE_CHILD", role)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let input = child.stdin.take().unwrap();
        let output = BufReader::new(child.stdout.take().unwrap());
        let mut process = Self {
            child,
            input,
            output,
        };
        process.input.write_all(&public.digest()).unwrap();
        process.send(packet.as_bytes());
        process.send(start.as_bytes());
        process
    }

    fn send(&mut self, bytes: &[u8]) {
        self.input
            .write_all(&(bytes.len() as u32).to_le_bytes())
            .unwrap();
        self.input.write_all(bytes).unwrap();
        self.input.flush().unwrap();
    }

    fn reply(&mut self) -> Vec<u8> {
        loop {
            let mut line = String::new();
            assert_ne!(
                self.output.read_line(&mut line).unwrap(),
                0,
                "child ended without reply"
            );
            if let Some((_, bytes)) = line.split_once("WIRE_REPLY=") {
                return unhex(bytes.trim());
            }
        }
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn independent_party_processes_import_publication_and_only_their_own_share() {
    independent_process_roundtrip(false);
}

#[test]
fn independent_seeded_party_processes_deliver_keys_only_to_one_and_verify_complete_window() {
    independent_process_roundtrip(true);
}

fn independent_process_roundtrip(seeded: bool) {
    let (matrix, table) = sources();
    let public = Arc::new(
        Publication::publish(
            [42; 32],
            &[Stage {
                id: ID,
                matrix: &matrix,
            }],
            &table,
        )
        .unwrap(),
    );
    let packet = public.export_wire().unwrap();
    let client = Client::new(Arc::clone(&public), public.digest()).unwrap();
    let mut window = if seeded {
        client.begin_seeded_wire([43; 32], &[Step::Lookup, Step::Linear(ID)])
    } else {
        client.begin_wire([43; 32], &[Step::Lookup, Step::Linear(ID)])
    }
    .unwrap();
    let starts = window.start_frames().unwrap();
    let mut zero = Process::start("0", &public, &packet, &starts[0]);
    let mut one = Process::start("1", &public, &packet, &starts[1]);
    let requests = window.lookup(1).unwrap();
    zero.send(requests[0].as_bytes());
    one.send(requests[1].as_bytes());
    let row = window.accept_replies(&zero.reply(), &one.reply()).unwrap();
    assert_eq!(
        row.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [1.5f32.to_bits(), (-0.0f32).to_bits()]
    );
    let inputs = [
        encode_input(row.values[0] as f64).unwrap(),
        encode_input(row.values[1] as f64).unwrap(),
    ];
    let requests = window.linear(&inputs).unwrap();
    zero.send(requests[0].as_bytes());
    one.send(requests[1].as_bytes());
    assert_eq!(
        window
            .accept_replies(&zero.reply(), &one.reply())
            .unwrap()
            .signed,
        &[9216]
    );
    assert!(zero.child.wait().unwrap().success());
    assert!(one.child.wait().unwrap().success());
    let accepted = window.finish().unwrap();
    assert_eq!(accepted.signed_linear_output(), &[9216]);
    assert_eq!(
        accepted.sharing_profile(),
        if seeded {
            SharingProfile::Aes128WindowV1
        } else {
            SharingProfile::FullShares
        }
    );
}
