```markdown
# Gollm Custom Provider Registration Fix (v0.1.9)

## Problem Description

When attempting to use a custom provider (e.g., "cerebras") registered using the documented method for custom providers (`providers.GetDefaultRegistry().Register("cerebras", NewCerebrasProvider)` within an `init()` function), the application failed during runtime when creating the LLM instance.

The following error occurred when calling `gollm.NewLLM(config.SetProvider("cerebras"), ...)`:

```go
ERROR: Failed to create internal LLM [error unknown provider: cerebras] ... failed to create or configure gollm LLM with provider cerebras: failed to create internal LLM: unknown provider: cerebras
```

This happened despite log messages confirming that the provider registration in the `init()` function was executed successfully before the `gollm.NewLLM` call.

## Root Cause Analysis

Upon inspecting the source code of `github.com/teilomillet/gollm` version `v0.1.9` (specifically in `gollm.go`), it was discovered that the top-level `gollm.NewLLM` function contained an issue related to provider registry handling.

While custom providers are correctly registered to the default singleton registry via `providers.GetDefaultRegistry().Register()`, the `gollm.NewLLM` function in v0.1.9 incorrectly performed the provider lookup using a *newly created* registry instance:

```go
// Incorrect line in gollm@v0.1.9/gollm.go within NewLLM function
provider, err := providers.NewProviderRegistry().Get(cfg.Provider, cfg.APIKeys[cfg.Provider], cfg.Model, cfg.ExtraHeaders)
```

This newly created registry (`providers.NewProviderRegistry()`) did *not* contain the custom providers registered with the default singleton registry, leading to the "unknown provider" error during the `Get` call.

(Note: A similar issue existed on the line calling the internal `llm.NewLLM` function, which also passed a new registry instead of the default one).

## Solution

The issue was resolved by modifying the gollm library code to consistently use the default provider registry singleton (`providers.GetDefaultRegistry()`) for all provider lookups and internal LLM creation within the `gollm.NewLLM` function.

The necessary changes involved:

1.  **Forking the Repository:** Created a fork of `github.com/teilomillet/gollm`.

2.  **Modifying `gollm.go`:** In the forked repository, located the `gollm.NewLLM` function in `gollm.go` and changed the problematic lines to use `providers.GetDefaultRegistry()`:

    ```go
    --- a/gollm.go
    +++ b/gollm.go
    @@ -2,7 +2,7 @@
     		cfg.ExtraHeaders["anthropic-beta"] = "prompt-caching-2024-07-31"
     	}
 
    -	baseLLM, err := llm.NewLLM(cfg, logger, providers.NewProviderRegistry())
    +	baseLLM, err := llm.NewLLM(cfg, logger, providers.GetDefaultRegistry()) // Use default registry
     	if err != nil {
     		logger.Error("Failed to create internal LLM", "error", err)
     		return nil, fmt.Errorf("failed to create internal LLM: %w", err)
 
    -	provider, err := providers.NewProviderRegistry().Get(cfg.Provider, cfg.APIKeys[cfg.Provider], cfg.Model, cfg.ExtraHeaders)
    +	provider, err := providers.GetDefaultRegistry().Get(cfg.Provider, cfg.APIKeys[cfg.Provider], cfg.Model, cfg.ExtraHeaders) // Use default registry
     	if err != nil {
     		return nil, fmt.Errorf("failed to get provider: %w", err)
     	}
    ```

3.  **Tagging the Fork:** Created a tag (e.g., `v0.1.1-fix-1`) in the forked repository pointing to the commit containing the fix.

## Implementation in Project

To utilize the fixed version of the library, the project's `go.mod` file was updated with a `replace` directive pointing to the tagged commit in the forked repository:

```go
module FIG_Inference

go 1.23.1

// Replace the original gollm dependency with the fixed fork
replace github.com/teilomillet/gollm => github.com/guiperry/gollm_cerebras v0.1.1-fix-1

require (
	// ... other dependencies ...
	github.com/teilomillet/gollm v0.1.9 // Original version requirement remains
	// ...
)

// ... indirect dependencies ...
```

After adding the `replace` directive, running `go mod tidy` ensures that the project uses the code from the specified fork, resolving the "unknown provider" error and allowing the custom "cerebras" provider to be correctly instantiated and used.
```