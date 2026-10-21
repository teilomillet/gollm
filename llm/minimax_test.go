package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/providers"
	"github.com/teilomillet/gollm/utils"
)

type minimaxRoundTripper struct {
	t *testing.T
}

func (rt minimaxRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	rt.t.Helper()
	require.Equal(rt.t, http.MethodPost, req.Method)
	require.Equal(rt.t, "api.minimax.io", req.URL.Host)
	require.Equal(rt.t, "/v1/chat/completions", req.URL.Path)
	require.Equal(rt.t, "Bearer test-api-key", req.Header.Get("Authorization"))
	require.Equal(rt.t, "application/json", req.Header.Get("Content-Type"))

	body, err := io.ReadAll(req.Body)
	require.NoError(rt.t, err)
	var payload struct {
		Model    string `json:"model"`
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	require.NoError(rt.t, json.Unmarshal(body, &payload))
	require.Equal(rt.t, providers.MiniMaxModelM31Flash, payload.Model)
	require.Equal(rt.t, []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}{{Role: "user", Content: "ping"}}, payload.Messages)

	responseBody := `{"choices":[{"message":{"role":"assistant","content":"pong"}}]}`
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(responseBody)),
		Request:    req,
	}, nil
}

func TestMiniMaxM31FlashGenerateUsesChatBoundary(t *testing.T) {
	cfg := config.NewConfig()
	cfg.Provider = "minimax"
	cfg.Model = providers.MiniMaxModelM31Flash
	cfg.APIKeys["minimax"] = "test-api-key"
	cfg.MaxRetries = 0
	cfg.RetryDelay = 0
	cfg.Timeout = time.Second

	internal, err := NewLLM(cfg, utils.NewLogger(utils.LogLevelError), providers.GetDefaultRegistry())
	require.NoError(t, err)
	impl := internal.(*LLMImpl)
	impl.client.Transport = minimaxRoundTripper{t: t}

	response, err := impl.Generate(context.Background(), &Prompt{Input: "ping"})
	require.NoError(t, err)
	require.Equal(t, "pong", response)
}
