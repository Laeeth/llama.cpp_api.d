extern(C):
enum LLAMA_FILE_VERSION =1;
enum LLAMA_FILE_MAGIC= 0x67676a74; // 'ggjt' in hex
enum LLAMA_FILE_MAGIC_UNVERSIONED =0x67676d6c; // pre-versioned files


//
// C interface
//
// TODO: show sample usage
//

struct llama_context;

alias llama_token = int;

struct llama_token_data {
	llama_token id;  // token id
	float p;     // probability of the token
	float plog;  // log probability of the token
}

alias llama_progress_callback = void function(float progress, void *ctx);

struct llama_context_params {
        int n_ctx;   // text context
        int n_parts; // -1 for default
        int seed;    // RNG seed, 0 for random

        bool f16_kv;     // use fp16 for KV cache
        bool logits_all; // the llama_eval() call computes all logits, not just the last one
        bool vocab_only; // only load the vocabulary, no weights
        bool use_mmap;   // use mmap if possible
        bool use_mlock;  // force system to keep model in RAM
        bool embedding;  // embedding mode only

        // called with a progress value between 0 and 1, pass NULL to disable
        llama_progress_callback progress_callback;
        // context pointer passed to the progress callback
        void * progress_callback_user_data;
    }

    // model file types
    enum llama_ftype
	{
        LLAMA_FTYPE_ALL_F32     = 0,
        LLAMA_FTYPE_MOSTLY_F16  = 1,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
        LLAMA_FTYPE_MOSTLY_Q4_2 = 5,  // except 1d tensors
    }

    llama_context_params llama_context_default_params();

    bool llama_mmap_supported();
    bool llama_mlock_supported();

    // Various functions for loading a ggml llama model.
    // Allocate (almost) all memory needed for the model.
    // Return NULL on failure
    llama_context*  llama_init_from_file( const(char)*  path_model,
            llama_context_params   params);

    // Frees all allocated memory
    void llama_free(llama_context*  ctx);

    // TODO: not great API - very likely to change
    // Returns 0 on success
    int llama_model_quantize(
            const(char)*  fname_inp,
            const(char)*  fname_out,
      		llama_ftype   ftype);

    // Apply a LoRA adapter to a loaded model
    // path_base_model is the path to a higher quality model to use as a base for
    // the layers modified by the adapter. Can be NULL to use the current loaded model.
    // The model needs to be reloaded before applying a new adapter, otherwise the adapter
    // will be applied on top of the previous one
    // Returns 0 on success
    int llama_apply_lora_from_file(
            llama_context*  ctx,
                      const(char)*  path_lora,
                      const(char)*  path_base_model,
                             int   n_threads);

    // Returns the KV cache that will contain the context for the
    // ongoing prediction with the model.
    const(ubyte)* llama_get_kv_cache(llama_context*  ctx);

    // Returns the size of the KV cache
    size_t llama_get_kv_cache_size(llama_context*  ctx);

    // Returns the number of tokens in the KV cache
    int llama_get_kv_cache_token_count(llama_context*  ctx);

    // Sets the KV cache containing the current context for the model
    void llama_set_kv_cache( llama_context*  ctx,
                   const(ubyte)* kv_cache,
                          size_t   n_size,
                             int   n_token_count);

    // Run the llama inference to obtain the logits and probabilities for the next token.
    // tokens + n_tokens is the provided batch of new tokens to process
    // n_past is the number of tokens to use from previous eval calls
    // Returns 0 on success
    int llama_eval( llama_context*  ctx,
               const(llama_token)* tokens,
                             int   n_tokens,
                             int   n_past,
                             int   n_threads);

    // Convert the provided text into tokens.
    // The tokens pointer must be large enough to hold the resulting tokens.
    // Returns the number of tokens on success, no more than n_max_tokens
    // Returns a negative number on failure - the number of tokens that would have been returned
    // TODO: not sure if correct
    int llama_tokenize(
            llama_context*  ctx,
                      const(char)*  text,
                     llama_token * tokens,
                             int   n_max_tokens,
                            bool   add_bos);

    int llama_n_vocab(llama_context*  ctx);
    int llama_n_ctx  (llama_context*  ctx);
    int llama_n_embd (llama_context*  ctx);

    // Token logits obtained from the last call to llama_eval()
    // The logits for the last token are stored in the last row
    // Can be mutated in order to change the probabilities of the next token
    // Rows: n_tokens
    // Cols: n_vocab
    float*  llama_get_logits(llama_context*  ctx);

    // Get the embeddings for the input
    // shape: [n_embd] (1-dimensional)
    float*  llama_get_embeddings(llama_context*  ctx);

    // Token Id -> String. Uses the vocabulary in the provided context
    const(char)*  llama_token_to_str(llama_context*  ctx, llama_token token);

    // Special tokens
    llama_token llama_token_bos();
    llama_token llama_token_eos();

    // TODO: improve the last_n_tokens interface ?
    llama_token llama_sample_top_p_top_k(
       llama_context* ctx,
          const llama_token * last_n_tokens_data,
                        int   last_n_tokens_size,
                        int   top_k,
                      float   top_p,
                      float   temp,
                      float   repeat_penalty);

    // Performance information
    void llama_print_timings(llama_context*  ctx);
    void llama_reset_timings(llama_context*  ctx);

    // Print system information
    const(char)*  llama_print_system_info();

struct ggml_tensor;

//std::vector<std::pair<std::string, struct ggml_tensor *>>& llama_internal_get_tensor_map(llama_context*  ctx);

