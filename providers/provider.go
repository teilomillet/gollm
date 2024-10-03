package providers

type Provider interface {
	Name() string
	Endpoint() string
	Headers() map[string]string
	PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error)
	ParseResponse(body []byte) (string, error)
	SetExtraHeaders(extraHeaders map[string]string)
	HandleFunctionCalls(body []byte) ([]byte, error)
}
