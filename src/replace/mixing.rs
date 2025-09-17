use crate::circuit::circuit::{CircuitSeq};
use crate::replace::replace::{obfuscate, compress};
use std::fs::File;
use std::io::Write;
use rusqlite::Connection;

fn obfuscate_and_compress(c: &CircuitSeq, conn: &Connection) -> CircuitSeq {
    // Step 1: Obfuscate circuit, get positions of inverses
    let (mut final_circuit, inverse_starts) = obfuscate(c);
    println!("{:?} Obf Len: {}", pin_counts(&final_circuit), final_circuit.gates.len());
    // Step 2: For each gate, compress its "inverse+gate+next_random" slice
    // Reverse iteration to avoid index shifting issues
    for i in (0..c.gates.len()).rev() {
        // ri^-1 start
        let start = inverse_starts[i];

        // r_{i+1} start is the next inverse start
        let end = inverse_starts[i + 1]; // safe because i < c.gates.len()
        // Slice the subcircuit: r_i^-1 ⋅ g_i ⋅ r_{i+1}
        let sub_slice = &final_circuit.gates[start..end];

        // Wrap it into a CircuitSeq
        let sub_circuit = CircuitSeq {
            gates: sub_slice.to_vec(),
        };

        // Compress the subcircuit
        let compressed_sub = compress(&sub_circuit, 1, conn);

        // Replace the slice in the final circuit
        final_circuit.gates.splice(start..end, compressed_sub.gates);
    }
    println!("{:?} Compressed Len: {}", pin_counts(&final_circuit), final_circuit.gates.len());
    final_circuit
}

pub fn pin_counts(circuit: &CircuitSeq) -> Vec<usize> {
    let num_wires = circuit.num_wires();
    let mut counts = vec![0; num_wires];
    for gate in &circuit.gates {
        counts[gate[0] as usize] += 1;
        counts[gate[1] as usize] += 1;
        counts[gate[2] as usize] += 1;
    }
    counts
}

pub fn main_mix(c: &CircuitSeq, rounds: usize, conn: &Connection) {
    // Start with the input circuit
    let mut circuit = c.clone();

    // Repeat obfuscate + compress 'rounds' times
    for _ in 0..rounds {
        circuit = obfuscate_and_compress(&circuit, conn);
    }

    // Convert the final circuit to string
    let circuit_str = circuit.to_circuit().to_string();

    // Write to file
    let mut file = File::create("circuit.txt").expect("Failed to create file");
    file.write_all(circuit_str.as_bytes())
        .expect("Failed to write circuit to file");

    println!("Final circuit written to circuit.txt");
}