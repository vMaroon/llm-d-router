package main

import (
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	pb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

func main() {
	address := os.Args[1]
	// Test-only client: the isolated EPP generates an ephemeral self-signed certificate.
	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(credentials.NewTLS(&tls.Config{InsecureSkipVerify: true})), grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(16<<20)))
	if err != nil {
		panic(err)
	}
	defer conn.Close()
	client := pb.NewExternalProcessorClient(conn)
	cases := []struct {
		name, path string
		body       map[string]any
	}{
		{"openai-short", "/v1/chat/completions", map[string]any{"messages": []any{map[string]any{"role": "user", "content": "Say ready."}}}},
		{"anthropic-tools", "/v1/messages", map[string]any{
			"messages":    []any{map[string]any{"role": "user", "content": "Call ping."}},
			"tools":       []any{map[string]any{"name": "ping", "description": "A test tool", "input_schema": map[string]any{"type": "object", "properties": map[string]any{}}}},
			"tool_choice": map[string]any{"type": "tool", "name": "ping"},
		}},
		{"anthropic-tool-result", "/v1/messages", map[string]any{"messages": []any{
			map[string]any{"role": "user", "content": "Call ping."},
			map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "tool_use", "id": "toolu_probe", "name": "ping", "input": map[string]any{}}}},
			map[string]any{"role": "user", "content": []any{map[string]any{"type": "tool_result", "tool_use_id": "toolu_probe", "content": "pong"}}},
		}}},
		{"anthropic-long", "/v1/messages", map[string]any{"messages": []any{map[string]any{"role": "user", "content": strings.Repeat("hello ", 240000)}}}},
	}
	if len(os.Args) > 2 && os.Args[2] == "short" {
		cases = cases[:3]
	}
	for _, tc := range cases {
		tc.body["model"] = "zai-org/GLM-5.3"
		tc.body["max_tokens"] = 32000
		tc.body["stream"] = true
		data, _ := json.Marshal(tc.body)
		ctx, cancel := context.WithTimeout(context.Background(), 40*time.Second)
		start := time.Now()
		stream, err := client.Process(ctx)
		if err != nil {
			panic(err)
		}
		headers := []*core.HeaderValue{}
		for k, v := range map[string]string{":method": "POST", ":path": tc.path, ":authority": "isolated-render-canary", "content-type": "application/json", "x-request-id": "render-discovery-" + tc.name} {
			headers = append(headers, &core.HeaderValue{Key: k, RawValue: []byte(v)})
		}
		err = stream.Send(&pb.ProcessingRequest{Request: &pb.ProcessingRequest_RequestHeaders{RequestHeaders: &pb.HttpHeaders{Headers: &core.HeaderMap{Headers: headers}}}})
		if err != nil {
			panic(err)
		}
		err = stream.Send(&pb.ProcessingRequest{Request: &pb.ProcessingRequest_RequestBody{RequestBody: &pb.HttpBody{Body: data, EndOfStream: true}}})
		if err != nil {
			panic(err)
		}
		result := map[string]any{"case": tc.name, "request_bytes": len(data)}
		var forwardedBody []byte
		for {
			resp, err := stream.Recv()
			if err != nil {
				result["error"] = err.Error()
				break
			}
			if immediate := resp.GetImmediateResponse(); immediate != nil {
				result["immediate_response"] = immediate.String()
				break
			}
			if h := resp.GetRequestHeaders(); h != nil {
				result["headers"] = h.Response.GetHeaderMutation().String()
			}
			if b := resp.GetRequestBody(); b != nil {
				body := b.Response.GetBodyMutation().GetBody()
				done := true
				if sr := b.Response.GetBodyMutation().GetStreamedResponse(); sr != nil {
					body = sr.Body
					done = sr.EndOfStream
				}
				forwardedBody = append(forwardedBody, body...)
				if !done {
					continue
				}
				if len(forwardedBody) > 0 {
					sum := sha256.Sum256(forwardedBody)
					result["forwarded_sha256"] = hex.EncodeToString(sum[:])
					original := sha256.Sum256(data)
					result["identical_forwarded_body"] = sum == original
					result["forwarded_bytes"] = len(forwardedBody)
					var forwarded map[string]any
					if json.Unmarshal(forwardedBody, &forwarded) == nil {
						result["forwarded_max_tokens"] = forwarded["max_tokens"]
					}
				} else {
					result["body_unchanged"] = true
				}
				break
			}
		}
		result["seconds"] = time.Since(start).Seconds()
		stream.CloseSend()
		cancel()
		out, _ := json.Marshal(result)
		fmt.Println(string(out))
	}
}
