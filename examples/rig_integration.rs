// Example: Using Kimi ACP Bridge with Rig (Rust)
// 
// Add to Cargo.toml:
// [dependencies]
// rig = "0.5"
// tokio = { version = "1", features = ["full"] }
// anyhow = "1"

use rig::providers::openai;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Point to local bridge instead of OpenAI
    let client = openai::Client::from_url(
        "http://localhost:8080/v1",
        "dummy-api-key"  // Bridge ignores this
    );
    
    // Create an agent with Kimi K2.5
    let agent = client
        .agent("kimi-k2.5")
        .preamble("You are a helpful coding assistant.")
        .build();
    
    // Simple prompt
    let response = agent.prompt("Explain what this bridge does").await?;
    println!("Response:\n{}\n", response);
    
    // Multi-turn conversation
    let agent = client
        .agent("kimi-k2.5")
        .preamble("You are a helpful coding assistant.")
        .build();
    
    let response = agent
        .prompt("What is the capital of France?")
        .await?;
    println!("Q: What is the capital of France?");
    println!("A: {}\n", response);
    
    Ok(())
}
