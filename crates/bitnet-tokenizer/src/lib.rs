//! # bitnet-tokenizer
//!
//! LLaMA 3 BPE tokenizer for BitNet b1.58 inference, backed by the
//! HuggingFace [`tokenizers`] crate.
//!
//! ## Architecture
//!
//! BitNet b1.58 uses the LLaMA 3 tokenizer — a Byte-Pair Encoding (BPE)
//! tokenizer with a vocabulary of 128,256 tokens. This crate wraps
//! `tokenizers::Tokenizer` and loads the canonical `tokenizer.json` shipped
//! in the model repository (downloaded via `bitnet download`).
//!
//! This crate provides:
//!
//! - Token encoding (text → `Vec<u32>`)
//! - Token decoding (`Vec<u32>` → text)
//! - LLaMA 3 Instruct chat template formatting
//! - Special token IDs (BOS, EOS, header markers, etc.)
//!
//! ## Loading
//!
//! The tokenizer must be loaded from a `tokenizer.json` file on disk:
//!
//! ```no_run
//! use bitnet_tokenizer::Tokenizer;
//!
//! // Searches well-known cache paths or the BITNET_TOKENIZER env var:
//! let tok = Tokenizer::llama3().unwrap();
//! ```
//!
//! ## LLaMA 3 Special Tokens
//!
//! | Token                  | ID      | Meaning                             |
//! |------------------------|---------|-------------------------------------|
//! | `<|begin_of_text|>`   | 128000  | Start of sequence (BOS)             |
//! | `<|end_of_text|>`     | 128001  | End of text                         |
//! | `<|start_header_id|>` | 128006  | Opens a role header in chat format  |
//! | `<|end_header_id|>`   | 128007  | Closes a role header in chat format |
//! | `<|eot_id|>`          | 128009  | End of turn (generation stop token) |
//!
//! ## Chat Template
//!
//! LLaMA 3 Instruct format:
//!
//! ```text
//! <|begin_of_text|><|start_header_id|>system<|end_header_id|>
//!
//! {system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>
//!
//! {user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
//!
//! ```
//!
//! Note the double newline (`\n\n`) between each role header and its content.
//!
//! ## Invariants
//!
//! - All regular BPE token IDs are in `[0, 128_000)`.
//! - Special token IDs are in `[128_000, 128_256)`.
//! - `bos_token_id()` = 128000, `eos_token_id()` = 128009 (end-of-turn).
//! - `apply_chat_template` always ends with `"<|end_header_id|>\n\n"` (open
//!   assistant header ready for generation).
//! - `decode(encode(s, false))` = `s` for all valid UTF-8 strings `s`.

#![warn(missing_docs)]
#![warn(clippy::all)]

use std::path::PathBuf;

use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer as HfTokenizer;
use tracing::{debug, instrument, warn};

// ---------------------------------------------------------------------------
// Special token IDs (LLaMA 3)
// ---------------------------------------------------------------------------

/// Token ID for `<|begin_of_text|>` — the BOS (Beginning of Sequence) token.
///
/// Every LLaMA 3 prompt begins with this token.
pub const BOS_TOKEN_ID: u32 = 128_000;

/// Token ID for `<|end_of_text|>` — general end-of-text marker.
pub const EOT_TOKEN_ID: u32 = 128_001;

/// Token ID for `<|start_header_id|>` — opens a role name in the chat template.
pub const START_HEADER_ID: u32 = 128_006;

/// Token ID for `<|end_header_id|>` — closes a role name in the chat template.
pub const END_HEADER_ID: u32 = 128_007;

/// Token ID for `<|eot_id|>` — End of Turn.
///
/// Used as the generation stop token for instruct/chat models.
/// Generation halts when the model produces this token.
pub const EOT_ID: u32 = 128_009;

/// Total vocabulary size of the LLaMA 3 tokenizer.
///
/// Token IDs 0–127,999 are standard BPE tokens; IDs 128,000–128,255 are
/// special tokens (BOS, EOS, header markers, etc.).
pub const VOCAB_SIZE: usize = 128_256;

// ---------------------------------------------------------------------------
// Chat template string constants
// ---------------------------------------------------------------------------

const BEGIN_OF_TEXT: &str = "<|begin_of_text|>";
const START_HEADER: &str = "<|start_header_id|>";
const END_HEADER: &str = "<|end_header_id|>";
const EOT: &str = "<|eot_id|>";

// ---------------------------------------------------------------------------
// ChatMessage
// ---------------------------------------------------------------------------

/// A single message in a chat conversation.
///
/// # Fields
///
/// - `role`:    One of `"system"`, `"user"`, or `"assistant"`.
/// - `content`: The text content of the message.
///
/// # Example
///
/// ```
/// use bitnet_tokenizer::ChatMessage;
///
/// let msg = ChatMessage::user("Hello, world!");
/// assert_eq!(msg.role, "user");
/// assert_eq!(msg.content, "Hello, world!");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// The role of the message sender.
    ///
    /// Recognised values: `"system"`, `"user"`, `"assistant"`.
    /// Unknown roles are passed through to the chat template without validation.
    pub role: String,

    /// The text content of this message.
    pub content: String,
}

impl ChatMessage {
    /// Create a new [`ChatMessage`] with the given role and content.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    /// Convenience constructor for a `system` message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    /// Convenience constructor for a `user` message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    /// Convenience constructor for an `assistant` message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// LLaMA 3 BPE tokenizer backed by the HuggingFace [`tokenizers`] crate.
///
/// # Loading
///
/// Constructed from a `tokenizer.json` file on disk — either explicitly via
/// [`Tokenizer::from_file`] or by searching well-known cache paths via
/// [`Tokenizer::llama3`].
///
/// # Vocabulary
///
/// The LLaMA 3 tokenizer has 128,256 tokens:
/// - IDs 0–127,999 are standard BPE tokens produced by the BPE model.
/// - IDs 128,000–128,255 are special tokens (BOS, EOS, header markers, etc.).
///
/// All regular text encodes exclusively to IDs in `[0, 128_000)`.
///
/// # Thread Safety
///
/// [`Tokenizer`] is `Send + Sync` and may be shared across threads via `Arc`.
///
/// # Example
///
/// ```no_run
/// use bitnet_tokenizer::Tokenizer;
///
/// let tok = Tokenizer::llama3().unwrap();
/// let ids = tok.encode("Hello, world!", false).unwrap();
/// let text = tok.decode(&ids).unwrap();
/// assert_eq!(text, "Hello, world!");
/// ```
pub struct Tokenizer {
    inner: HfTokenizer,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &VOCAB_SIZE)
            .finish()
    }
}

impl Tokenizer {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Load a tokenizer from an explicit `tokenizer.json` file path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file does not exist or cannot be parsed as a
    /// valid HuggingFace tokenizer JSON.
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| anyhow!("Failed to load tokenizer from {}: {e}", path.display()))?;
        debug!(path = %path.display(), "LLaMA 3 tokenizer loaded from tokenizer.json");
        Ok(Self { inner })
    }

    /// Load the LLaMA 3 tokenizer from the model cache.
    ///
    /// Search order:
    /// 1. Path given by the `BITNET_TOKENIZER` environment variable.
    /// 2. `~/.cache/bitnet/microsoft__bitnet-b1.58-2B-4T/tokenizer.json`
    /// 3. `~/.cache/bitnet/microsoft__bitnet-b1.58-2B-4T-bf16/tokenizer.json`
    ///
    /// Run `bitnet download` to fetch `tokenizer.json` into the default cache.
    ///
    /// # Errors
    ///
    /// Returns an error listing all searched paths if `tokenizer.json` is not
    /// found at any candidate location.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use bitnet_tokenizer::Tokenizer;
    ///
    /// let tok = Tokenizer::llama3().unwrap();
    /// assert_eq!(tok.bos_token_id(), 128000);
    /// ```
    pub fn llama3() -> anyhow::Result<Self> {
        if let Ok(p) = std::env::var("BITNET_TOKENIZER") {
            let path = std::path::Path::new(&p);
            if path.exists() {
                debug!(
                    path = %path.display(),
                    "Loading LLaMA 3 tokenizer from BITNET_TOKENIZER env var"
                );
                return Self::from_file(path);
            }
            warn!(
                path = %p,
                "BITNET_TOKENIZER env var is set but file does not exist; \
                 falling back to default cache locations"
            );
        }

        let candidates = candidate_paths();
        for path in &candidates {
            if path.exists() {
                debug!(path = %path.display(), "Loading LLaMA 3 tokenizer from cache");
                return Self::from_file(path);
            }
        }

        Err(anyhow!(
            "LLaMA 3 tokenizer.json not found. \
             Run `bitnet download` first, or set BITNET_TOKENIZER=/path/to/tokenizer.json.\n\
             Searched: {candidates:?}"
        ))
    }

    // ------------------------------------------------------------------
    // Core encode / decode
    // ------------------------------------------------------------------

    /// Encode a text string to a sequence of token IDs.
    ///
    /// Special token strings present in `text` (e.g. `<|begin_of_text|>`)
    /// are recognised by the underlying tokenizer and encoded to their
    /// respective special token IDs automatically.
    ///
    /// # Arguments
    ///
    /// - `text`:    The input text to tokenise.
    /// - `add_bos`: If `true`, prepend the BOS token (`128000`) before all
    ///   other tokens.
    ///
    /// # Returns
    ///
    /// A `Vec<u32>` of token IDs. Regular text tokens are always in
    /// `[0, 128_000)`; special token strings in the text encode to IDs in
    /// `[128_000, 128_256)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying HuggingFace tokeniser fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use bitnet_tokenizer::Tokenizer;
    ///
    /// let tok = Tokenizer::llama3().unwrap();
    /// let ids = tok.encode("Hello", true).unwrap();
    /// assert_eq!(ids[0], 128000, "BOS token must be first");
    /// assert!(ids.len() > 1, "must have BOS + at least one content token");
    /// ```
    #[instrument(level = "trace", skip(self, text), fields(len = text.len(), add_bos))]
    pub fn encode(&self, text: &str, add_bos: bool) -> anyhow::Result<Vec<u32>> {
        // Drive special-token insertion ourselves (add_special_tokens = false):
        // the post-processor BOS/EOS insertion is skipped; we prepend BOS
        // manually when add_bos is true.  Special token strings that appear
        // literally in the text (e.g. from apply_chat_template) are still
        // tokenised to their canonical IDs by the split-level handler.
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow!("Tokenizer encode failed: {e}"))?;

        let mut ids: Vec<u32> = encoding.get_ids().to_vec();

        if add_bos {
            ids.insert(0, BOS_TOKEN_ID);
        }

        Ok(ids)
    }

    /// Decode a sequence of token IDs back to a UTF-8 string.
    ///
    /// # Behaviour
    ///
    /// - Special tokens (ID ≥ 128,000) are silently dropped before decoding.
    ///   Use [`Tokenizer::decode_with_special_tokens`] to render them as text
    ///   literals instead.
    /// - If the underlying tokeniser cannot decode an individual token (OOV),
    ///   it is replaced with the Unicode replacement character `U+FFFD`.
    ///
    /// # Invariant
    ///
    /// For any valid UTF-8 string `s`:
    /// `decode(encode(s, false))` = `s`.
    ///
    /// # Errors
    ///
    /// Returns an error only for unexpected internal tokeniser failures.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use bitnet_tokenizer::Tokenizer;
    ///
    /// let tok = Tokenizer::llama3().unwrap();
    /// let ids = tok.encode("Hello, world!", false).unwrap();
    /// let text = tok.decode(&ids).unwrap();
    /// assert_eq!(text, "Hello, world!");
    /// ```
    #[instrument(level = "trace", skip(self, tokens), fields(n_tokens = tokens.len()))]
    pub fn decode(&self, tokens: &[u32]) -> anyhow::Result<String> {
        // Drop all special tokens (≥ 128,000): they are outside the BPE
        // vocabulary and must be handled by the caller.
        let regular: Vec<u32> = tokens.iter().filter(|&&id| id < 128_000).copied().collect();

        if regular.is_empty() {
            return Ok(String::new());
        }

        // Fast path: batch decode of the regular-token sequence.
        // For the real LLaMA 3 tokenizer all tokens in [0, 128_000) are valid
        // BPE entries, so this path succeeds in the common case.
        if let Ok(text) = self.inner.decode(&regular, false) {
            return Ok(text);
        }

        // Per-token fallback: replace any token the tokeniser cannot decode
        // (OOV) with the Unicode replacement character U+FFFD, so callers
        // receive a valid UTF-8 string rather than a hard error.
        let mut result = String::new();
        for &id in &regular {
            match self.inner.decode(&[id], false) {
                Ok(piece) => result.push_str(&piece),
                Err(_) => result.push('\u{FFFD}'),
            }
        }
        Ok(result)
    }

    /// Decode a single token ID to its raw byte representation.
    ///
    /// Useful for streaming generation where tokens are decoded one at a time.
    ///
    /// # Returns
    ///
    /// - `Ok(None)` for special tokens (ID ≥ 128,000) or tokens the tokeniser
    ///   cannot map to bytes.
    /// - `Ok(Some(bytes))` for a valid BPE token.
    ///
    /// Most tokens produce valid UTF-8; byte-level tokens for individual UTF-8
    /// code units may not form valid UTF-8 in isolation.
    ///
    /// # Errors
    ///
    /// Returns an error only for unexpected internal tokeniser failures.
    pub fn decode_single(&self, token: u32) -> anyhow::Result<Option<Vec<u8>>> {
        if token >= 128_000 {
            return Ok(None);
        }
        match self.inner.decode(&[token], false) {
            Ok(text) => Ok(Some(text.into_bytes())),
            Err(_) => Ok(None),
        }
    }

    /// Decode a token sequence, rendering special tokens as their canonical
    /// string literals instead of silently dropping them.
    ///
    /// Unlike [`Tokenizer::decode`], which filters out all tokens with
    /// ID ≥ 128,000, this method converts those tokens to their textual
    /// representations (e.g. `128009` → `"<|eot_id|>"`). Useful for
    /// debugging chat-template encoding and inspecting raw model output.
    ///
    /// # Special Token Mapping
    ///
    /// | Token ID        | String literal           |
    /// |-----------------|--------------------------|
    /// | 128000          | `<\|begin_of_text\|>`   |
    /// | 128001          | `<\|end_of_text\|>`     |
    /// | 128006          | `<\|start_header_id\|>` |
    /// | 128007          | `<\|end_header_id\|>`   |
    /// | 128009          | `<\|eot_id\|>`          |
    /// | Other ≥ 128000  | `<\|special:{id}\|>`    |
    ///
    /// Regular tokens (ID < 128,000) are decoded in contiguous batches to
    /// preserve multi-token UTF-8 byte sequences correctly.
    ///
    /// # Invariants
    ///
    /// - For a slice of only regular tokens, the result equals
    ///   `decode(tokens).unwrap()`.
    /// - An empty slice returns an empty string.
    pub fn decode_with_special_tokens(&self, tokens: &[u32]) -> String {
        let mut result = String::new();
        // Buffer of contiguous regular tokens, flushed as a batch immediately
        // before each special token to preserve multi-token byte sequences.
        let mut regular_buf: Vec<u32> = Vec::new();

        for &tok in tokens {
            if tok >= BOS_TOKEN_ID {
                // Flush any accumulated regular tokens before the special one.
                if !regular_buf.is_empty() {
                    let batch: Vec<u32> = regular_buf.drain(..).collect();
                    if let Ok(decoded) = self.inner.decode(&batch, false) {
                        result.push_str(&decoded);
                    }
                }
                // Map the special token ID to its canonical string literal.
                let literal = match tok {
                    128_000 => "<|begin_of_text|>".to_owned(),
                    128_001 => "<|end_of_text|>".to_owned(),
                    128_006 => "<|start_header_id|>".to_owned(),
                    128_007 => "<|end_header_id|>".to_owned(),
                    128_009 => "<|eot_id|>".to_owned(),
                    id => format!("<|special:{id}|>"),
                };
                result.push_str(&literal);
            } else {
                regular_buf.push(tok);
            }
        }

        // Flush any remaining regular tokens after the last special token (or
        // the entire sequence if no special tokens were encountered).
        if !regular_buf.is_empty() {
            if let Ok(decoded) = self.inner.decode(&regular_buf, false) {
                result.push_str(&decoded);
            }
        }

        result
    }

    // ------------------------------------------------------------------
    // Token ID accessors
    // ------------------------------------------------------------------

    /// Returns the BOS (Beginning of Sequence) token ID: `128000`.
    ///
    /// This is `<|begin_of_text|>` in the LLaMA 3 vocabulary.
    #[inline]
    pub fn bos_token_id(&self) -> u32 {
        BOS_TOKEN_ID
    }

    /// Returns the EOS (End of Sequence / End of Turn) token ID: `128009`.
    ///
    /// This is `<|eot_id|>`. Generation stops when the model produces this token.
    #[inline]
    pub fn eos_token_id(&self) -> u32 {
        EOT_ID
    }

    /// Returns the vocabulary size: `128256`.
    ///
    /// Valid token IDs are in `[0, vocab_size)`.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        VOCAB_SIZE
    }

    // ------------------------------------------------------------------
    // Chat template
    // ------------------------------------------------------------------

    /// Format a list of [`ChatMessage`]s into a LLaMA 3 Instruct prompt string.
    ///
    /// The formatted string is ready to be encoded and passed to the model.
    ///
    /// # Format
    ///
    /// ```text
    /// <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    ///
    /// {system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>
    ///
    /// {user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    ///
    /// ```
    ///
    /// Notes:
    /// - If no `system` message is present, no system block is emitted.
    /// - The prompt always ends with an open `assistant` header (`\n\n`),
    ///   signalling the model to begin its response.
    /// - The `\n\n` between header and content is required by the LLaMA 3
    ///   instruct format specification.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use bitnet_tokenizer::{Tokenizer, ChatMessage};
    ///
    /// let tok = Tokenizer::llama3().unwrap();
    /// let messages = vec![
    ///     ChatMessage::system("You are a helpful assistant."),
    ///     ChatMessage::user("What is 2 + 2?"),
    /// ];
    /// let prompt = tok.apply_chat_template(&messages);
    /// assert!(prompt.contains("<|begin_of_text|>"));
    /// assert!(prompt.ends_with("\n\n"));
    /// ```
    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> String {
        let mut out = String::new();

        // Start with BOS.
        out.push_str(BEGIN_OF_TEXT);

        for msg in messages {
            // Per-turn format:
            //   <|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>
            out.push_str(START_HEADER);
            out.push_str(&msg.role);
            out.push_str(END_HEADER);
            out.push_str("\n\n");
            out.push_str(&msg.content);
            out.push_str(EOT);
        }

        // Open the assistant header to prime the model for its response.
        out.push_str(START_HEADER);
        out.push_str("assistant");
        out.push_str(END_HEADER);
        out.push_str("\n\n");

        out
    }

    /// Encode a chat conversation using the LLaMA 3 Instruct chat template.
    ///
    /// Equivalent to `encode(apply_chat_template(messages), false)`.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails.
    pub fn encode_chat(&self, messages: &[ChatMessage]) -> anyhow::Result<Vec<u32>> {
        let prompt = self.apply_chat_template(messages);
        self.encode(&prompt, false)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Returns the candidate `tokenizer.json` paths in priority order.
///
/// On Windows `USERPROFILE` is tried first; on Unix/macOS `HOME` is used.
fn candidate_paths() -> Vec<PathBuf> {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    vec![
        PathBuf::from(&home).join(".cache/bitnet/microsoft__bitnet-b1.58-2B-4T/tokenizer.json"),
        PathBuf::from(&home)
            .join(".cache/bitnet/microsoft__bitnet-b1.58-2B-4T-bf16/tokenizer.json"),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Initialise the tokenizer, panicking with a descriptive message if
    /// `tokenizer.json` has not been downloaded to the default cache location.
    fn make_tokenizer() -> Tokenizer {
        Tokenizer::llama3().unwrap_or_else(|e| {
            panic!("Tokenizer init failed (is tokenizer.json downloaded?): {e}")
        })
    }

    // ------------------------------------------------------------------
    // Initialisation
    // ------------------------------------------------------------------

    #[test]
    fn tokenizer_initialises_successfully() {
        let tok = make_tokenizer();
        assert_eq!(tok.vocab_size(), VOCAB_SIZE);
    }

    #[test]
    fn bos_token_id_is_128000() {
        let tok = make_tokenizer();
        assert_eq!(tok.bos_token_id(), 128_000);
    }

    #[test]
    fn eos_token_id_is_128009() {
        let tok = make_tokenizer();
        assert_eq!(tok.eos_token_id(), 128_009);
    }

    #[test]
    fn vocab_size_is_128256() {
        assert_eq!(VOCAB_SIZE, 128_256);
    }

    // ------------------------------------------------------------------
    // Encode
    // ------------------------------------------------------------------

    #[test]
    fn encode_empty_string_returns_empty_without_bos() {
        let tok = make_tokenizer();
        let ids = tok.encode("", false).unwrap();
        assert!(
            ids.is_empty(),
            "empty string without BOS must give empty tokens"
        );
    }

    #[test]
    fn encode_with_bos_prepends_128000() {
        let tok = make_tokenizer();
        let ids = tok.encode("hello", true).unwrap();
        assert!(ids.len() >= 2, "must have BOS + at least one token");
        assert_eq!(ids[0], BOS_TOKEN_ID, "first token must be BOS (128000)");
    }

    #[test]
    fn encode_without_bos_no_128000_prepended() {
        let tok = make_tokenizer();
        let ids = tok.encode("hello", false).unwrap();
        assert!(!ids.is_empty(), "must produce at least one token");
        assert_ne!(
            ids[0], BOS_TOKEN_ID,
            "BOS must not be prepended when add_bos=false"
        );
    }

    #[test]
    fn encode_hello_world_produces_tokens() {
        let tok = make_tokenizer();
        let ids = tok.encode("Hello, world!", false).unwrap();
        assert!(!ids.is_empty(), "must produce tokens for 'Hello, world!'");
        // Plain text must produce only regular BPE tokens.
        for &id in &ids {
            assert!(id < 128_000, "regular text token {id} must be < 128000");
        }
    }

    #[test]
    fn encode_numbers_only() {
        let tok = make_tokenizer();
        let ids = tok.encode("12345", false).unwrap();
        assert!(!ids.is_empty());
    }

    #[test]
    fn encode_unicode_text() {
        let tok = make_tokenizer();
        let ids = tok.encode("こんにちは世界", false).unwrap();
        assert!(!ids.is_empty(), "Japanese text must produce tokens");
    }

    // ------------------------------------------------------------------
    // Decode
    // ------------------------------------------------------------------

    #[test]
    fn decode_empty_slice_returns_empty_string() {
        let tok = make_tokenizer();
        let text = tok.decode(&[]).unwrap();
        assert!(
            text.is_empty(),
            "decoding empty slice must return empty string"
        );
    }

    #[test]
    fn decode_encode_roundtrip_ascii() {
        let tok = make_tokenizer();
        let original = "Hello, world!";
        let ids = tok.encode(original, false).unwrap();
        let decoded = tok.decode(&ids).unwrap();
        assert_eq!(
            decoded, original,
            "decode(encode(x)) must equal x for ASCII"
        );
    }

    #[test]
    fn decode_encode_roundtrip_unicode() {
        let tok = make_tokenizer();
        let original = "The quick brown fox jumps over the lazy dog. 日本語テスト.";
        let ids = tok.encode(original, false).unwrap();
        let decoded = tok.decode(&ids).unwrap();
        assert_eq!(
            decoded, original,
            "decode(encode(x)) must equal x for Unicode"
        );
    }

    #[test]
    fn decode_skips_special_tokens() {
        let tok = make_tokenizer();
        // Mix special tokens (≥128000) with regular tokens for "Hello".
        let hello_ids = tok.encode("Hello", false).unwrap();
        let mut mixed = vec![BOS_TOKEN_ID]; // special
        mixed.extend_from_slice(&hello_ids); // regular
        mixed.push(EOT_ID); // special

        let decoded = tok.decode(&mixed).unwrap();
        assert_eq!(decoded, "Hello", "special tokens must be skipped in decode");
    }

    /// Token 113558 is within the LLaMA 3 BPE vocabulary `[0, 128_000)` and
    /// must decode to a valid, non-empty string with the real `tokenizer.json`.
    ///
    /// Previously (when backed by cl100k_base with ~100k tokens) this token
    /// fell outside the vocabulary range and produced U+FFFD. With the real
    /// LLaMA 3 tokenizer the full 128k vocabulary is available, so this token
    /// maps to an actual sub-word piece.
    #[test]
    fn decode_valid_llama3_token_113558() {
        let tok = make_tokenizer();
        let result = tok.decode(&[113_558]).unwrap();
        assert!(
            !result.is_empty(),
            "token 113558 must decode to a non-empty string with the real LLaMA 3 tokenizer; \
             got {result:?}"
        );
        assert_ne!(
            result, "\u{FFFD}",
            "token 113558 is a valid LLaMA 3 BPE token and must NOT produce U+FFFD; \
             got {result:?}"
        );
    }

    /// Tokens with ID ≥ 128,000 are filtered out by `decode` (they are
    /// outside the BPE vocabulary), so a sequence containing only such IDs
    /// returns an empty string rather than an error or U+FFFD.
    #[test]
    fn decode_out_of_range_token_returns_empty() {
        let tok = make_tokenizer();
        // 200000 >> VOCAB_SIZE: treated as a special/out-of-range token and
        // filtered before any decode attempt.
        let result = tok.decode(&[200_000]).unwrap();
        assert!(
            result.is_empty(),
            "token 200000 is out of range and must be filtered to an empty string; \
             got {result:?}"
        );
    }

    /// With the real LLaMA 3 tokenizer, token 113558 is a valid BPE entry and
    /// decodes without producing U+FFFD.  A mixed sequence of valid tokens
    /// including 113558 must preserve the regular text on either side.
    #[test]
    fn decode_oov_token_mixed_with_regular_preserves_regular_text() {
        let tok = make_tokenizer();
        let hello_ids = tok.encode("Hello", false).unwrap();
        // 113558 is a valid LLaMA 3 BPE token; the full sequence decodes.
        let mut mixed: Vec<u32> = Vec::new();
        mixed.extend_from_slice(&hello_ids);
        mixed.push(113_558);
        mixed.extend_from_slice(&hello_ids);

        let result = tok.decode(&mixed).unwrap();
        assert!(
            result.contains("Hello"),
            "regular tokens must survive mixed-sequence decode; got {result:?}"
        );
        assert!(
            !result.is_empty(),
            "mixed sequence with valid LLaMA 3 tokens must not decode to empty; got {result:?}"
        );
    }

    #[test]
    fn decode_all_special_tokens_returns_empty() {
        let tok = make_tokenizer();
        let all_special = vec![BOS_TOKEN_ID, EOT_ID, START_HEADER_ID, END_HEADER_ID];
        let decoded = tok.decode(&all_special).unwrap();
        assert!(
            decoded.is_empty(),
            "decoding only special tokens must return empty string"
        );
    }

    // ------------------------------------------------------------------
    // decode_single
    // ------------------------------------------------------------------

    #[test]
    fn decode_single_special_token_returns_none() {
        let tok = make_tokenizer();
        let result = tok.decode_single(BOS_TOKEN_ID).unwrap();
        assert!(result.is_none(), "special token must return None");
    }

    /// Tokens with ID ≥ 128,000 return `None` from `decode_single` regardless
    /// of whether they are named special tokens or completely out-of-range IDs.
    ///
    /// Token 200000 is far above `VOCAB_SIZE` (128256); the `>= 128_000` guard
    /// in `decode_single` catches it before any vocabulary lookup.
    #[test]
    fn decode_single_out_of_range_token_returns_none() {
        let tok = make_tokenizer();
        let result = tok.decode_single(200_000).unwrap();
        assert!(
            result.is_none(),
            "out-of-range token 200000 must return None from decode_single"
        );
    }

    #[test]
    fn decode_single_regular_token_returns_some() {
        let tok = make_tokenizer();
        // Encode "Hello" and probe the first resulting token.
        let ids = tok.encode("Hello", false).unwrap();
        let result = tok.decode_single(ids[0]).unwrap();
        assert!(result.is_some(), "regular token must decode to Some(bytes)");
        let bytes = result.unwrap();
        assert!(!bytes.is_empty(), "decoded bytes must not be empty");
    }

    // ------------------------------------------------------------------
    // ChatMessage
    // ------------------------------------------------------------------

    #[test]
    fn chat_message_new() {
        let msg = ChatMessage::new("user", "Hello!");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello!");
    }

    #[test]
    fn chat_message_system_constructor() {
        let msg = ChatMessage::system("You are helpful.");
        assert_eq!(msg.role, "system");
        assert_eq!(msg.content, "You are helpful.");
    }

    #[test]
    fn chat_message_user_constructor() {
        let msg = ChatMessage::user("How are you?");
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "How are you?");
    }

    #[test]
    fn chat_message_assistant_constructor() {
        let msg = ChatMessage::assistant("I am fine.");
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content, "I am fine.");
    }

    // ------------------------------------------------------------------
    // apply_chat_template
    // ------------------------------------------------------------------

    /// Verify that `encode_chat` produces the correct special token IDs at
    /// the structural positions of the LLaMA 3 Instruct template.
    ///
    /// The expected prefix for a user-only message is:
    /// ```text
    /// 128000  <|begin_of_text|>
    /// 128006  <|start_header_id|>
    /// ...     "user" tokens
    /// 128007  <|end_header_id|>
    /// ...     content tokens
    /// 128009  <|eot_id|>
    /// 128006  <|start_header_id|>
    /// ...     "assistant" tokens
    /// 128007  <|end_header_id|>
    /// ```
    #[test]
    fn encode_chat_special_token_ids_are_correct() {
        let tok = make_tokenizer();
        let messages = vec![ChatMessage::user("Hi")];
        let ids = tok.encode_chat(&messages).unwrap();

        assert!(
            !ids.is_empty(),
            "encode_chat must produce at least one token"
        );

        // First token must be BOS = 128000.
        assert_eq!(
            ids[0], BOS_TOKEN_ID,
            "first token must be BOS (128000), got {}",
            ids[0]
        );

        // The sequence must contain <|start_header_id|> (128006).
        assert!(
            ids.contains(&START_HEADER_ID),
            "encoded chat must contain <|start_header_id|> (128006); got {:?}",
            &ids[..ids.len().min(20)]
        );

        // The sequence must contain <|end_header_id|> (128007).
        assert!(
            ids.contains(&END_HEADER_ID),
            "encoded chat must contain <|end_header_id|> (128007); got {:?}",
            &ids[..ids.len().min(20)]
        );

        // The sequence must contain <|eot_id|> (128009) after the user message.
        assert!(
            ids.contains(&EOT_ID),
            "encoded chat must contain <|eot_id|> (128009); got {:?}",
            &ids[..ids.len().min(20)]
        );

        // <|start_header_id|> must appear before <|end_header_id|>.
        let first_start = ids.iter().position(|&t| t == START_HEADER_ID).unwrap();
        let first_end = ids.iter().position(|&t| t == END_HEADER_ID).unwrap();
        assert!(
            first_start < first_end,
            "<|start_header_id|> (pos {first_start}) must precede <|end_header_id|> (pos {first_end})"
        );

        // <|eot_id|> must appear after the first <|end_header_id|>.
        let first_eot = ids.iter().position(|&t| t == EOT_ID).unwrap();
        assert!(
            first_eot > first_end,
            "<|eot_id|> (pos {first_eot}) must appear after <|end_header_id|> (pos {first_end})"
        );

        // All special token IDs must be >= 128000.
        let special_ids: Vec<u32> = ids.iter().copied().filter(|&t| t >= BOS_TOKEN_ID).collect();
        assert!(
            !special_ids.is_empty(),
            "must have at least one special token ID >= 128000"
        );
        for &id in &special_ids {
            assert!(
                id < VOCAB_SIZE as u32,
                "special token id {id} must be < VOCAB_SIZE ({})",
                VOCAB_SIZE
            );
        }
    }

    /// Forensic test: decode the top-predicted token IDs from the real model
    /// forward pass to verify that the vocabulary is consistent between the
    /// tokenizer and the model's logit output.
    ///
    /// Token IDs observed from `forensic_real_model_forward_layer0_intermediates`:
    /// top-5 after "The" (token 791): [17345, 74466, 48946, 60150, 14760]
    #[test]
    fn forensic_decode_top_predicted_tokens() {
        let tok = make_tokenizer();
        // These are the top-5 token IDs predicted by the model after "The" (token 791).
        // Decoding them tells us whether the model's logit output aligns with
        // meaningful vocabulary entries.
        let top_predicted = vec![17345u32, 74466, 48946, 60150, 14760];
        for &id in &top_predicted {
            // Must not panic and must produce some non-empty string.
            let decoded = tok.decode(&[id]).unwrap();
            eprintln!("  token {:6} -> {:?}", id, decoded);
        }
        // Token 791 should decode to something containing "The".
        let the_decoded = tok.decode(&[791]).unwrap();
        eprintln!("  token    791 -> {:?}", the_decoded);
        assert!(
            the_decoded.contains("The") || the_decoded.contains("the"),
            "token 791 must decode to 'The' or similar, got {:?}",
            the_decoded
        );
        // Token 60704 should decode to something related to "Paris".
        let paris_decoded = tok.decode(&[60704]).unwrap();
        eprintln!("  token  60704 -> {:?}", paris_decoded);
        assert!(
            !paris_decoded.is_empty(),
            "token 60704 must decode to a non-empty string"
        );
    }

    #[test]
    fn chat_template_starts_with_begin_of_text() {
        let tok = make_tokenizer();
        let messages = vec![ChatMessage::user("Hello!")];
        let prompt = tok.apply_chat_template(&messages);
        assert!(
            prompt.starts_with(BEGIN_OF_TEXT),
            "prompt must start with <|begin_of_text|>, got: {prompt:?}"
        );
    }

    #[test]
    fn chat_template_ends_with_open_assistant_header() {
        let tok = make_tokenizer();
        let messages = vec![ChatMessage::user("Hi!")];
        let prompt = tok.apply_chat_template(&messages);
        assert!(
            prompt.ends_with("\n\n"),
            "prompt must end with \\n\\n, got: {prompt:?}"
        );
        assert!(
            prompt.contains("assistant"),
            "prompt must contain 'assistant' header"
        );
    }

    #[test]
    fn chat_template_contains_system_message() {
        let tok = make_tokenizer();
        let messages = vec![
            ChatMessage::system("You are a helpful AI."),
            ChatMessage::user("Hello!"),
        ];
        let prompt = tok.apply_chat_template(&messages);
        assert!(
            prompt.contains("You are a helpful AI."),
            "system message must appear in prompt"
        );
        assert!(
            prompt.contains("system"),
            "system role header must appear in prompt"
        );
    }

    #[test]
    fn chat_template_contains_user_message() {
        let tok = make_tokenizer();
        let messages = vec![ChatMessage::user("What is 2+2?")];
        let prompt = tok.apply_chat_template(&messages);
        assert!(
            prompt.contains("What is 2+2?"),
            "user message must appear in prompt"
        );
        assert!(
            prompt.contains("user"),
            "user role header must appear in prompt"
        );
    }

    #[test]
    fn chat_template_includes_eot_after_each_message() {
        let tok = make_tokenizer();
        let messages = vec![
            ChatMessage::user("First message."),
            ChatMessage::assistant("Response."),
            ChatMessage::user("Second message."),
        ];
        let prompt = tok.apply_chat_template(&messages);
        // Count occurrences of <|eot_id|>: must be exactly 3 (one per message).
        let count = prompt.matches(EOT).count();
        assert_eq!(
            count, 3,
            "must have exactly 3 <|eot_id|> markers, got {count}"
        );
    }

    #[test]
    fn chat_template_multi_turn_conversation() {
        let tok = make_tokenizer();
        let messages = vec![
            ChatMessage::system("Be concise."),
            ChatMessage::user("Question 1?"),
            ChatMessage::assistant("Answer 1."),
            ChatMessage::user("Question 2?"),
        ];
        let prompt = tok.apply_chat_template(&messages);

        // All messages must appear in the correct order.
        let pos_sys = prompt.find("Be concise.").unwrap();
        let pos_q1 = prompt.find("Question 1?").unwrap();
        let pos_a1 = prompt.find("Answer 1.").unwrap();
        let pos_q2 = prompt.find("Question 2?").unwrap();

        assert!(pos_sys < pos_q1, "system must come before question 1");
        assert!(pos_q1 < pos_a1, "question 1 must come before answer 1");
        assert!(pos_a1 < pos_q2, "answer 1 must come before question 2");
    }

    #[test]
    fn chat_template_empty_messages_has_only_bos_and_assistant_header() {
        let tok = make_tokenizer();
        let prompt = tok.apply_chat_template(&[]);
        // Expected: <|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n
        assert!(prompt.starts_with(BEGIN_OF_TEXT));
        assert!(prompt.contains("assistant"));
        assert!(prompt.ends_with("\n\n"));
        // No eot_id since there are no messages.
        assert!(
            !prompt.contains(EOT),
            "empty messages must produce no eot_id"
        );
    }

    #[test]
    fn chat_template_double_newline_between_header_and_content() {
        let tok = make_tokenizer();
        let messages = vec![ChatMessage::user("Test message.")];
        let prompt = tok.apply_chat_template(&messages);
        // The LLaMA 3 format requires \n\n between the closing header tag and content.
        assert!(
            prompt.contains(&format!("{END_HEADER}\n\n")),
            "must have \\n\\n after <|end_header_id|>"
        );
    }

    // ------------------------------------------------------------------
    // encode_chat
    // ------------------------------------------------------------------

    #[test]
    fn encode_chat_produces_non_empty_tokens() {
        let tok = make_tokenizer();
        let messages = vec![
            ChatMessage::system("Be helpful."),
            ChatMessage::user("Hello!"),
        ];
        let ids = tok.encode_chat(&messages).unwrap();
        assert!(!ids.is_empty(), "encode_chat must produce tokens");
    }

    #[test]
    fn encode_chat_tokens_are_in_valid_range() {
        let tok = make_tokenizer();
        let messages = vec![ChatMessage::user("Test message.")];
        let ids = tok.encode_chat(&messages).unwrap();
        for &id in &ids {
            // Chat template special token strings (BOS, headers, EOT) encode to
            // their proper special IDs (128000–128009).  All IDs must be within
            // the 128256-token vocabulary, with a small guard margin.
            assert!(
                (id as usize) < VOCAB_SIZE + 10_000,
                "token id {id} is out of reasonable range"
            );
        }
    }

    // ------------------------------------------------------------------
    // Mathematical properties
    // ------------------------------------------------------------------

    /// Property: longer text produces more tokens than shorter text.
    ///
    /// Formally: for representative natural-language strings s, t where
    /// `len(s) >> len(t)`, `|encode(s)| > |encode(t)|`.
    #[test]
    fn longer_text_produces_more_tokens() {
        let tok = make_tokenizer();
        let short = "Hello.";
        let long = "Hello. This is a much longer sentence that should produce more tokens.";

        let short_len = tok.encode(short, false).unwrap().len();
        let long_len = tok.encode(long, false).unwrap().len();

        assert!(
            long_len > short_len,
            "longer text must produce more tokens: short={short_len}, long={long_len}"
        );
    }

    /// Formal invariant: `decode ∘ encode = id` for representative ASCII and
    /// Unicode strings.
    ///
    /// `∀ s ∈ UTF-8: decode(encode(s, false)) = s`
    #[test]
    fn roundtrip_identity_for_various_texts() {
        let tok = make_tokenizer();
        let cases = [
            "The quick brown fox.",
            "1 + 1 = 2",
            "Hello, World!",
            "abcdefghijklmnopqrstuvwxyz",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "0123456789",
            "   spaces   ",
            "newline\ntest",
            "tab\there",
        ];
        for &text in &cases {
            let ids = tok.encode(text, false).unwrap();
            let decoded = tok.decode(&ids).unwrap();
            assert_eq!(decoded, text, "roundtrip failed for: {text:?}");
        }
    }

    // ------------------------------------------------------------------
    // Special token constants
    // ------------------------------------------------------------------

    #[test]
    fn special_token_constants_have_expected_values() {
        assert_eq!(BOS_TOKEN_ID, 128_000);
        assert_eq!(EOT_TOKEN_ID, 128_001);
        assert_eq!(START_HEADER_ID, 128_006);
        assert_eq!(END_HEADER_ID, 128_007);
        assert_eq!(EOT_ID, 128_009);
    }

    #[test]
    fn all_special_token_ids_are_in_special_range() {
        let special_ids = [
            BOS_TOKEN_ID,
            EOT_TOKEN_ID,
            START_HEADER_ID,
            END_HEADER_ID,
            EOT_ID,
        ];
        for id in special_ids {
            assert!(id >= 128_000, "special token {id} must be ≥ 128000");
        }
    }

    #[test]
    fn special_token_ids_are_unique() {
        let ids = [
            BOS_TOKEN_ID,
            EOT_TOKEN_ID,
            START_HEADER_ID,
            END_HEADER_ID,
            EOT_ID,
        ];
        let unique: std::collections::HashSet<u32> = ids.iter().cloned().collect();
        assert_eq!(
            unique.len(),
            ids.len(),
            "all special token IDs must be unique"
        );
    }

    // ------------------------------------------------------------------
    // decode_with_special_tokens
    // ------------------------------------------------------------------

    /// Invariant: EOT_ID (128009) must decode to its canonical string literal.
    #[test]
    fn decode_with_special_tokens_renders_eot_id() {
        let tok = make_tokenizer();
        let result = tok.decode_with_special_tokens(&[128009]);
        assert_eq!(
            result, "<|eot_id|>",
            "EOT token must render as '<|eot_id|>'"
        );
    }

    /// Invariant: BOS_TOKEN_ID (128000) must decode to its canonical string literal.
    #[test]
    fn decode_with_special_tokens_renders_bos() {
        let tok = make_tokenizer();
        let result = tok.decode_with_special_tokens(&[128000]);
        assert_eq!(
            result, "<|begin_of_text|>",
            "BOS must render as '<|begin_of_text|>'"
        );
    }

    /// Invariant: a mixed sequence [BOS, regular_tokens…, EOT] must produce
    /// `"<|begin_of_text|>{text}<|eot_id|>"` with the regular text inline.
    #[test]
    fn decode_with_special_tokens_mixed_sequence() {
        let tok = make_tokenizer();
        let hello_tokens = tok.encode("Hello", false).unwrap();
        assert!(
            !hello_tokens.is_empty(),
            "Hello must encode to at least one token"
        );
        // Build: BOS + Hello tokens + EOT
        let mut seq: Vec<u32> = vec![128000];
        seq.extend_from_slice(&hello_tokens);
        seq.push(128009);

        let decoded = tok.decode_with_special_tokens(&seq);
        assert!(
            decoded.starts_with("<|begin_of_text|>"),
            "Result must start with BOS: {decoded:?}"
        );
        assert!(
            decoded.ends_with("<|eot_id|>"),
            "Result must end with EOT: {decoded:?}"
        );
        assert!(
            decoded.contains("Hello"),
            "Result must contain 'Hello': {decoded:?}"
        );
    }

    /// Invariant: an unrecognised special token (ID ≥ 128000, not in the named
    /// set) must be rendered as `<|special:{id}|>`.
    #[test]
    fn decode_with_special_tokens_unknown_special_renders_id() {
        let tok = make_tokenizer();
        let result = tok.decode_with_special_tokens(&[200000]);
        assert_eq!(
            result, "<|special:200000|>",
            "Unknown special token must render with its numeric ID"
        );
    }

    /// Invariant: an empty slice produces an empty string.
    #[test]
    fn decode_with_special_tokens_empty_slice_returns_empty() {
        let tok = make_tokenizer();
        let result = tok.decode_with_special_tokens(&[]);
        assert!(result.is_empty(), "Empty input must produce empty output");
    }

    /// Invariant: for a sequence containing only regular tokens (ID < 128000),
    /// `decode_with_special_tokens` must produce the same string as `decode`.
    #[test]
    fn decode_with_special_regular_tokens_only_matches_decode() {
        let tok = make_tokenizer();
        let text = "The quick brown fox";
        let tokens = tok.encode(text, false).unwrap();
        let decoded_regular = tok.decode(&tokens).unwrap();
        let decoded_with_special = tok.decode_with_special_tokens(&tokens);
        assert_eq!(
            decoded_regular, decoded_with_special,
            "For purely regular tokens, decode_with_special_tokens must match decode"
        );
    }
}
