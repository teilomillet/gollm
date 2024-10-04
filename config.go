// File: config.go

package gollm

import (
	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/utils"
)

// Re-export types
type (
	Config       = config.Config
	ConfigOption = config.ConfigOption
	LogLevel     = utils.LogLevel
	MemoryOption = config.MemoryOption
)

// Re-export functions
var (
	LoadConfig   = config.LoadConfig
	ApplyOptions = config.ApplyOptions
)

// Re-export ConfigOption functions
var (
	SetProvider         = config.SetProvider
	SetModel            = config.SetModel
	SetOllamaEndpoint   = config.SetOllamaEndpoint
	SetTemperature      = config.SetTemperature
	SetMaxTokens        = config.SetMaxTokens
	SetTopP             = config.SetTopP
	SetFrequencyPenalty = config.SetFrequencyPenalty
	SetPresencePenalty  = config.SetPresencePenalty
	SetTimeout          = config.SetTimeout
	SetMaxRetries       = config.SetMaxRetries
	SetRetryDelay       = config.SetRetryDelay
	SetLogLevel         = config.SetLogLevel
	SetSeed             = config.SetSeed
	SetMinP             = config.SetMinP
	SetRepeatPenalty    = config.SetRepeatPenalty
	SetRepeatLastN      = config.SetRepeatLastN
	SetMirostat         = config.SetMirostat
	SetMirostatEta      = config.SetMirostatEta
	SetMirostatTau      = config.SetMirostatTau
	SetTfsZ             = config.SetTfsZ
	SetExtraHeaders     = config.SetExtraHeaders
	SetEnableCaching    = config.SetEnableCaching
	SetMemory           = config.SetMemory
	SetAPIKey           = config.SetAPIKey
	NewConfig           = config.NewConfig
)

// Constants
const (
	LogLevelOff   = utils.LogLevelOff
	LogLevelError = utils.LogLevelError
	LogLevelWarn  = utils.LogLevelWarn
	LogLevelInfo  = utils.LogLevelInfo
	LogLevelDebug = utils.LogLevelDebug
)
