#[tokio::main]
async fn main() -> anyhow::Result<()> {
    llm_bench::run_bench_command(std::env::args().skip(1).collect()).await
}
