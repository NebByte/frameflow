# Gemini vision wired, without an API key -- 2026-08-22

The vision half of the reasoning layer now runs on this machine's Application
Default Credentials through Vertex AI. Nothing is pasted, nothing is stored,
nothing needs rotating, and the token renews itself.

## Live result

One frame of the café clip at 640x360, asked what sits just outside the frame:

```
- person's body (left)
- table edge (left)
- person's back (right)
- more seating (right)
```

Those land as `asserted`, never `measured` -- a model is a party making claims
about a place it was not present at, held to the same standard as a script page.
They cannot move the honesty number.

## Why not an API key

The credential a user has to hand is usually an `AQ.` OAuth access token, which
dies inside an hour: both of the ones pasted into this project were already
expired by the time they were tried (HTTP 401). Google also began rejecting
unrestricted API keys in June 2026; new AI Studio keys are scoped "auth keys".

ADC sidesteps all of it. `gcloud auth application-default print-access-token`
mints a fresh bearer token on demand, so a long render cannot fail halfway
through on an expired credential.

## The two failures on the way, and what they meant

1. **`generativelanguage` + ADC token -> HTTP 403, "insufficient authentication
   scopes."** The ADC created for the Colab CLI carries cloud-platform,
   colaboratory and drive.file, not the generative-language scope.
2. **Vertex + ADC -> HTTP 403, "Agent Platform API has not been used in project
   moonlit-app-9060 before or it is disabled."** A real, single-switch blocker
   rather than a credential problem. Enabled with:

       gcloud services enable aiplatform.googleapis.com --project=moonlit-app-9060

   After which the same request returned HTTP 200.

## What changed in the code

`gemini.adc_token()` mints and briefly caches a bearer token.
`gemini.adc_project()` finds the project ADC bills against.
`GeminiVision.endpoint()` returns the Vertex path when running keyless and the
public API path when handed a key, so both routes stay supported.
`credential()` prefers an explicit key and falls back to ADC.

The old test asserted "no token means no call". That contract changed: no token
no longer means no credential, so it now asserts on having nothing at all.
`test_auth_route` pins which endpoint each credential implies.

36 assertions in test_gemini.py; 309 across the suite.
