// File: utils/mcp_tool.go
package utils

import (
	"encoding/json"
	"fmt"
)

// MCPTool represents a Tool in the Model Context Protocol format
type MCPTool struct {
	Type        string                 `json:"type"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// ToMCP converts a Tool to its MCP representation
func (t *Tool) ToMCP() (*MCPTool, error) {
	return &MCPTool{
		Type:        t.Type,
		Name:        t.Function.Name,
		Description: t.Function.Description,
		Parameters:  t.Function.Parameters,
	}, nil
}

// FromMCP creates a Tool from its MCP representation
func FromMCP(mcp *MCPTool) (*Tool, error) {
	if mcp == nil {
		return nil, fmt.Errorf("cannot convert nil MCP tool")
	}

	return &Tool{
		Type: mcp.Type,
		Function: Function{
			Name:        mcp.Name,
			Description: mcp.Description,
			Parameters:  mcp.Parameters,
		},
	}, nil
}

// MarshalJSON implements the json.Marshaler interface
func (t *Tool) MarshalJSON() ([]byte, error) {
	mcp, err := t.ToMCP()
	if err != nil {
		return nil, fmt.Errorf("failed to convert to MCP format: %w", err)
	}
	return json.Marshal(mcp)
}

// UnmarshalJSON implements the json.Unmarshaler interface
func (t *Tool) UnmarshalJSON(data []byte) error {
	var mcp MCPTool
	if err := json.Unmarshal(data, &mcp); err != nil {
		return fmt.Errorf("failed to unmarshal MCP format: %w", err)
	}

	tool, err := FromMCP(&mcp)
	if err != nil {
		return fmt.Errorf("failed to convert from MCP format: %w", err)
	}

	*t = *tool
	return nil
}
