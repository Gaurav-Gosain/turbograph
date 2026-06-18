package storage

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

// S3Config configures an S3-compatible object store.
type S3Config struct {
	Endpoint  string // e.g. https://s3.us-east-1.amazonaws.com or http://localhost:9000
	Bucket    string
	Region    string // e.g. us-east-1
	AccessKey string
	SecretKey string
	// Prefix is an optional key prefix (a folder within the bucket).
	Prefix string
}

// S3 is a Blob backed by an S3-compatible service. It uses path-style addressing
// and SigV4 request signing, implemented on the standard library so no AWS SDK is
// required. It works with AWS S3, MinIO, Cloudflare R2, and similar services.
type S3 struct {
	cfg  S3Config
	base *url.URL
	http *http.Client
	now  func() time.Time
}

// NewS3 creates an S3 blob from cfg.
func NewS3(cfg S3Config) (*S3, error) {
	if cfg.Endpoint == "" || cfg.Bucket == "" || cfg.AccessKey == "" || cfg.SecretKey == "" {
		return nil, fmt.Errorf("storage: s3 requires endpoint, bucket, access key, and secret key")
	}
	if cfg.Region == "" {
		cfg.Region = "us-east-1"
	}
	u, err := url.Parse(strings.TrimRight(cfg.Endpoint, "/"))
	if err != nil {
		return nil, err
	}
	return &S3{cfg: cfg, base: u, http: &http.Client{Timeout: 2 * time.Minute}, now: time.Now}, nil
}

func (s *S3) objectURL(key string) string {
	full := key
	if s.cfg.Prefix != "" {
		full = strings.TrimRight(s.cfg.Prefix, "/") + "/" + key
	}
	return s.base.String() + "/" + s.cfg.Bucket + "/" + awsURIEncode(full, true)
}

// Put uploads data under key.
func (s *S3) Put(ctx context.Context, key string, data []byte) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, s.objectURL(key), bytes.NewReader(data))
	if err != nil {
		return err
	}
	s.sign(req, data)
	resp, err := s.http.Do(req)
	if err != nil {
		return err
	}
	return drain(resp, "put "+key)
}

// Get downloads the object under key.
func (s *S3) Get(ctx context.Context, key string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.objectURL(key), nil)
	if err != nil {
		return nil, err
	}
	s.sign(req, nil)
	resp, err := s.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return nil, ErrNotExist
	}
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, fmt.Errorf("storage: get %s: %s: %s", key, resp.Status, msg)
	}
	return io.ReadAll(resp.Body)
}

// Delete removes the object under key.
func (s *S3) Delete(ctx context.Context, key string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, s.objectURL(key), nil)
	if err != nil {
		return err
	}
	s.sign(req, nil)
	resp, err := s.http.Do(req)
	if err != nil {
		return err
	}
	if resp.StatusCode == http.StatusNotFound {
		resp.Body.Close()
		return ErrNotExist
	}
	return drain(resp, "delete "+key)
}

// List returns object keys (with the configured prefix stripped) that start with
// prefix.
func (s *S3) List(ctx context.Context, prefix string) ([]string, error) {
	q := url.Values{}
	q.Set("list-type", "2")
	full := prefix
	if s.cfg.Prefix != "" {
		full = strings.TrimRight(s.cfg.Prefix, "/") + "/" + prefix
	}
	if full != "" {
		q.Set("prefix", full)
	}
	u := s.base.String() + "/" + s.cfg.Bucket + "/?" + q.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	s.sign(req, nil)
	resp, err := s.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, fmt.Errorf("storage: list: %s: %s", resp.Status, msg)
	}
	var result struct {
		Contents []struct {
			Key string `xml:"Key"`
		} `xml:"Contents"`
	}
	if err := xml.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	trim := ""
	if s.cfg.Prefix != "" {
		trim = strings.TrimRight(s.cfg.Prefix, "/") + "/"
	}
	var keys []string
	for _, c := range result.Contents {
		keys = append(keys, strings.TrimPrefix(c.Key, trim))
	}
	sort.Strings(keys)
	return keys, nil
}

func drain(resp *http.Response, what string) error {
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return fmt.Errorf("storage: %s: %s: %s", what, resp.Status, msg)
	}
	io.Copy(io.Discard, resp.Body)
	return nil
}

// sign adds SigV4 authentication headers for the request and payload.
func (s *S3) sign(req *http.Request, payload []byte) {
	signV4(req, payload, s.now().UTC(), s.cfg.Region, "s3", s.cfg.AccessKey, s.cfg.SecretKey)
}

// signV4 signs an HTTP request with AWS Signature Version 4. It mutates req to add
// the x-amz-date, x-amz-content-sha256, and Authorization headers. The set of
// signed headers is host plus any x-amz-*, range, or content-type header present.
func signV4(req *http.Request, payload []byte, t time.Time, region, service, accessKey, secretKey string) {
	sum := sha256.Sum256(payload)
	payloadHash := hex.EncodeToString(sum[:])
	amzDate := t.Format("20060102T150405Z")
	dateStamp := t.Format("20060102")

	req.Header.Set("X-Amz-Date", amzDate)
	req.Header.Set("X-Amz-Content-Sha256", payloadHash)
	if req.Host == "" {
		req.Host = req.URL.Host
	}

	// Collect headers to sign.
	type hdr struct{ k, v string }
	var hs []hdr
	hs = append(hs, hdr{"host", req.URL.Host})
	for k, vals := range req.Header {
		lk := strings.ToLower(k)
		if strings.HasPrefix(lk, "x-amz-") || lk == "range" || lk == "content-type" {
			hs = append(hs, hdr{lk, strings.TrimSpace(strings.Join(vals, ","))})
		}
	}
	sort.Slice(hs, func(i, j int) bool { return hs[i].k < hs[j].k })
	var canonHeaders strings.Builder
	var signedNames []string
	for _, h := range hs {
		canonHeaders.WriteString(h.k)
		canonHeaders.WriteByte(':')
		canonHeaders.WriteString(h.v)
		canonHeaders.WriteByte('\n')
		signedNames = append(signedNames, h.k)
	}
	signedHeaders := strings.Join(signedNames, ";")

	canonicalURI := awsURIEncode(req.URL.Path, true)
	if canonicalURI == "" {
		canonicalURI = "/"
	}
	canonicalQuery := canonicalQueryString(req.URL.Query())

	canonicalRequest := strings.Join([]string{
		req.Method, canonicalURI, canonicalQuery, canonHeaders.String(), signedHeaders, payloadHash,
	}, "\n")
	crSum := sha256.Sum256([]byte(canonicalRequest))

	scope := dateStamp + "/" + region + "/" + service + "/aws4_request"
	stringToSign := strings.Join([]string{
		"AWS4-HMAC-SHA256", amzDate, scope, hex.EncodeToString(crSum[:]),
	}, "\n")

	kDate := hmacSHA256([]byte("AWS4"+secretKey), dateStamp)
	kRegion := hmacSHA256(kDate, region)
	kService := hmacSHA256(kRegion, service)
	kSigning := hmacSHA256(kService, "aws4_request")
	signature := hex.EncodeToString(hmacSHA256(kSigning, stringToSign))

	req.Header.Set("Authorization", fmt.Sprintf(
		"AWS4-HMAC-SHA256 Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		accessKey, scope, signedHeaders, signature))
}

func hmacSHA256(key []byte, data string) []byte {
	h := hmac.New(sha256.New, key)
	h.Write([]byte(data))
	return h.Sum(nil)
}

// canonicalQueryString builds the SigV4 canonical query string: sorted, with keys
// and values URI-encoded.
func canonicalQueryString(q url.Values) string {
	keys := make([]string, 0, len(q))
	for k := range q {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var parts []string
	for _, k := range keys {
		vals := q[k]
		sort.Strings(vals)
		for _, v := range vals {
			parts = append(parts, awsURIEncode(k, false)+"="+awsURIEncode(v, false))
		}
	}
	return strings.Join(parts, "&")
}

// awsURIEncode applies RFC 3986 encoding as S3 SigV4 requires: unreserved
// characters are left alone, everything else is percent-encoded, and a slash is
// preserved only when keepSlash is set (for path components).
func awsURIEncode(s string, keepSlash bool) string {
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c >= 'A' && c <= 'Z', c >= 'a' && c <= 'z', c >= '0' && c <= '9',
			c == '-', c == '_', c == '.', c == '~':
			b.WriteByte(c)
		case c == '/' && keepSlash:
			b.WriteByte('/')
		default:
			fmt.Fprintf(&b, "%%%02X", c)
		}
	}
	return b.String()
}
