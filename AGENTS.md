# Agent Instructions for gollm

## Build & Test Commands

```bash
# Build all packages
go build ./...

# Run all tests
go test ./...

# Run tests for specific package
go test github.com/teilomillet/gollm/llm -v

# Run specific test
go test -run TestMemoryGetMessages github.com/teilomillet/gollm/llm -v
```

## Git Workflow

- **Rebase before merging**: Always rebase feature branches onto main before merging to keep a clean, linear history. Use `git rebase main` or squash commits with `git rebase -i` to consolidate related changes.
- **Squash feedback fixes**: When addressing PR feedback, consider squashing fix commits into the original feature commits before merging.
- **Delete merged branches**: Clean up feature branches after merging.

## Code Style

- Follow Go conventions and existing patterns in the codebase
- Use interfaces for extensibility (e.g., `MemoryCapable` over concrete types)
- Write comprehensive tests, skip gracefully when providers unavailable
- Deep copy data structures to prevent shared state issues

## Testing Notes

- Tests using `lmstudio` provider will skip if LM Studio is not running
- Tests requiring API keys (Anthropic, OpenAI) will skip if keys are not set
- Use `createTestLLM` helper pattern to skip tests gracefully
