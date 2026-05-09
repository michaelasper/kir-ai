# Tool Parser Parity

`llm-tool-parser` keeps a vLLM-name routing table so model manifests and
operator configuration can use familiar parser names while the runtime routes to
the smallest compatible parser implementation.

## Supported vLLM Names

These vLLM parser names are accepted today:

| vLLM parser name | Kir parser |
| --- | --- |
| `deepseek_v3`, `deepseek_v31`, `deepseekv31`, `deepseek_v32`, `deepseekv32`, `deepseek_v4`, `deepseekv4` | `deep_seek` |
| `functiongemma`, `gemma4` | `gemma` |
| `hermes` | `hermes` |
| `qwen3coder`, `qwen3xml` | `qwen` |
| `xlam` | `xlam` |
| `json`, `openai`, `mistral`, `granite`, `granite_20b_fc`, `hunyuan_a13b`, `kimi_k2`, `minimax`, `minimax_m2`, `olmo3`, `phi4mini`, `seed_oss`, `step3`, `step3p5` | `json` |

The `json` parser accepts direct OpenAI-style objects, arrays of tool calls,
`tool_calls` wrapper objects, and stringified OpenAI `function.arguments`.
The `xlam` parser adds support for vLLM-style `[TOOL_CALLS]` markers, JSON code
fences, and `<tool_call>...</tool_call>` blocks.

Names with incompatible grammars stay unsupported until their output format is
implemented directly. `glm4_moe` is intentionally not routed to a generic
parser because vLLM documents it as an XML/incremental-string parser family,
which should not be treated as generic JSON.
