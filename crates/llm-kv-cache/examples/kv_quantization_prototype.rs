use llm_kv_cache::prototype_quantization::{
    evaluate_kv_quantization_fixture, tiny_gemma_value_fixture, tiny_qwen_value_fixture,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "| fixture | scheme | bits | rotation | payload bytes | memory ratio | recon mse | attention mse | decode ops | decode ns |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for fixture in [tiny_qwen_value_fixture(), tiny_gemma_value_fixture()] {
        let report = evaluate_kv_quantization_fixture(&fixture)?;
        for row in report.rows() {
            println!(
                "| {} | {:?} | {} | {} | {} | {:.5} | {:.8} | {:.8} | {} | {} |",
                report.fixture_model_family(),
                row.scheme(),
                row.bits().bit_width(),
                row.uses_rotation(),
                row.payload_bytes(),
                row.payload_memory_ratio(),
                row.reconstruction_mse(),
                row.attention_output_mse(),
                row.decode_estimated_ops(),
                row.decode_average_nanos()
            );
        }
    }
    Ok(())
}
