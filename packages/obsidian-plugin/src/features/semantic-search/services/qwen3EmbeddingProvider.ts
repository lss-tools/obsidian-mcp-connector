// Model: onnx-community/Qwen3-Embedding-0.6B-ONNX (decoder-based embedder, 1024d).
import type { BackendKind, EmbeddingProvider } from "../types";
import type { PipelineFactory } from "./embedder";
import { createTransformersProvider } from "./transformersProvider";

/**
 * Quantization per backend, pinned explicitly: left to its device
 * default, Transformers.js would pull fp32 (2.4 GB) on WebGPU.
 *
 * Measured against fp32 on 80 heading→section pairs of a Russian vault
 * (cosine to the fp32 vector / top-1 retrieval, fp32 itself 0.850):
 *   fp16  1.2 GB  0.9997 / 0.838   — and the fastest on WebGPU
 *   q4f16 567 MB  0.927  / 0.838   — 2-3x slower on WebGPU (MatMulNBits)
 *   q4    914 MB  0.927  / 0.825
 *   int8  614 MB  0.778  / 0.750
 * fp16 is float16 end to end, so it is WebGPU-only; WASM takes q4.
 */
export const QWEN3_DTYPE: Record<BackendKind, string> = {
  wasm: "q4",
  webgpu: "fp16",
};

// Qwen3-Embedding is instruction-aware on the query side only; documents
// are embedded as-is. One-sentence task taken from the model card.
const QUERY_INSTRUCTION =
  "Given a web search query, retrieve relevant passages that answer the query";

export function createQwen3EmbeddingProvider(
  pipelineFactory: PipelineFactory,
): EmbeddingProvider {
  return createTransformersProvider({
    modelId: "onnx-community/Qwen3-Embedding-0.6B-ONNX",
    providerKey: "qwen3-embedding-0.6b",
    dimensions: 1024,
    // Chunk budget (whitespace words, see chunker.countTokens). Kept at
    // 512 on both backends: the BPE vocabulary spends 3-4 tokens per
    // Cyrillic word, so a 512-word chunk already fills the real-token
    // bound below.
    maxInputTokensByBackend: { wasm: 512, webgpu: 512 },
    // The tokenizer declares model_max_length 131072; untruncated, one
    // long chunk makes attention memory explode (28 layers, seq²).
    tokenizerMaxLengthByBackend: { wasm: 1024, webgpu: 2048 },
    // The export is a decoder: besides last_hidden_state it emits 56
    // KV-cache tensors. Fetching only the hidden state took a short
    // query from ~285 ms to ~30 ms on WebGPU.
    fetchOutputs: ["last_hidden_state"],
    modelSizeBytes: 1_200_000_000,
    pooling: "last_token",
    taskPrompt: (text, role) =>
      role === "query"
        ? "Instruct: " + QUERY_INSTRUCTION + "\nQuery:" + text
        : text,
    pipelineFactory,
    // Same reasoning as EmbeddingGemma: padding to the longest text in
    // the batch makes attention memory batch × seq².
    batchSize: 4,
  });
}
